//! The exported C functions, driven the way a C caller drives them, over real QUIC on this
//! machine's loopback (IPv4: the crate never touches IPv6 unless asked for an IPv6 host).

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::thread::sleep;
use std::time::{Duration, Instant};

use mid_net_quic::*;

fn wait_for(what: &str, ms: u64, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(ms) {
        if cond() {
            return;
        }
        sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for: {what}");
}

struct Pair {
    server: *mut MidNetQuicServer,
    client: *mut MidNetQuicConn,
    served: *mut MidNetQuicConn, // the server's end of the same connection
}

fn open_server() -> *mut MidNetQuicServer {
    let host = CString::new("127.0.0.1").unwrap();
    let mut server = ptr::null_mut();
    let st = unsafe { mid_net_quic_server_open(host.as_ptr(), 0, &mut server) };
    assert_eq!(st, 0, "server_open");
    assert!(!server.is_null());
    server
}

fn connected_pair() -> Pair {
    let server = open_server();
    let port = unsafe { mid_net_quic_server_port(server) };
    assert_ne!(port, 0);
    let host = CString::new("127.0.0.1").unwrap();
    let mut client = ptr::null_mut();
    assert_eq!(unsafe { mid_net_quic_connect(host.as_ptr(), port, &mut client) }, 0);
    let mut served = ptr::null_mut();
    wait_for("the client to connect", 5000, || unsafe { mid_net_quic_conn_state(client) } == MID_NET_QUIC_CONNECTED);
    wait_for("the server to accept", 5000, || unsafe { mid_net_quic_server_accept(server, &mut served) } == 1);
    assert!(!served.is_null());
    Pair { server, client, served }
}

fn close_pair(p: Pair) {
    unsafe {
        mid_net_quic_conn_close(p.client, 0);
        mid_net_quic_conn_close(p.served, 0);
        mid_net_quic_server_close(p.server);
    }
}

fn recv_dgram(c: *mut MidNetQuicConn) -> Vec<u8> {
    let mut out = Vec::new();
    wait_for("a datagram", 5000, || {
        let mut buf = [0u8; 2048];
        let n = unsafe { mid_net_quic_recv_datagram(c, buf.as_mut_ptr(), buf.len()) };
        if n > 0 {
            out = buf[..n as usize].to_vec();
            true
        } else {
            assert!(n >= 0, "recv_datagram returned {n}");
            false
        }
    });
    out
}

fn recv_rel(c: *mut MidNetQuicConn, cap: usize) -> (u32, Vec<u8>) {
    let mut out = (0u32, Vec::new());
    wait_for("a reliable message", 5000, || {
        let mut buf = vec![0u8; cap];
        let mut id = 0u32;
        let n = unsafe { mid_net_quic_recv_reliable(c, &mut id, buf.as_mut_ptr(), buf.len()) };
        if n > 0 {
            out = (id, buf[..n as usize].to_vec());
            true
        } else {
            assert!(n >= 0, "recv_reliable returned {n}");
            false
        }
    });
    out
}

#[test]
fn a_server_opens_on_a_free_port_and_two_servers_get_two_ports() {
    let a = open_server();
    let b = open_server();
    let (pa, pb) = unsafe { (mid_net_quic_server_port(a), mid_net_quic_server_port(b)) };
    assert!(pa != 0 && pb != 0 && pa != pb);
    unsafe {
        mid_net_quic_server_close(a);
        mid_net_quic_server_close(b);
    }
}

#[test]
fn nothing_waiting_means_accept_returns_zero() {
    let server = open_server();
    let mut conn = ptr::null_mut();
    assert_eq!(unsafe { mid_net_quic_server_accept(server, &mut conn) }, 0);
    assert!(conn.is_null());
    unsafe { mid_net_quic_server_close(server) };
}

#[test]
fn datagrams_go_both_ways_whole() {
    let p = connected_pair();
    let up: Vec<u8> = (0..120u8).collect();
    let down: Vec<u8> = (0..60u8).rev().collect();
    assert_eq!(unsafe { mid_net_quic_send_datagram(p.client, up.as_ptr(), up.len()) }, 0);
    assert_eq!(recv_dgram(p.served), up);
    assert_eq!(unsafe { mid_net_quic_send_datagram(p.served, down.as_ptr(), down.len()) }, 0);
    assert_eq!(recv_dgram(p.client), down);
    close_pair(p);
}

#[test]
fn an_empty_datagram_is_not_delivered_and_a_huge_one_is_refused() {
    let p = connected_pair();
    assert_eq!(unsafe { mid_net_quic_send_datagram(p.client, ptr::null(), 0) }, 0);
    let marker = [7u8, 8, 9];
    assert_eq!(unsafe { mid_net_quic_send_datagram(p.client, marker.as_ptr(), marker.len()) }, 0);
    assert_eq!(recv_dgram(p.served), marker.to_vec(), "the empty one must not appear before the marker");
    let huge = vec![1u8; 100_000];
    assert_eq!(
        unsafe { mid_net_quic_send_datagram(p.client, huge.as_ptr(), huge.len()) },
        MidNetQuicStatus::SendFailed as i32
    );
    close_pair(p);
}

#[test]
fn a_datagram_too_big_for_the_buffer_is_dropped_with_a_code() {
    let p = connected_pair();
    let data = vec![5u8; 100];
    assert_eq!(unsafe { mid_net_quic_send_datagram(p.client, data.as_ptr(), data.len()) }, 0);
    let mut small = [0u8; 10];
    let mut got = i32::MIN;
    wait_for("the datagram to arrive", 5000, || {
        got = unsafe { mid_net_quic_recv_datagram(p.served, small.as_mut_ptr(), small.len()) };
        got != 0
    });
    assert_eq!(got, MidNetQuicStatus::BufferTooSmall as i32);
    let mut buf = [0u8; 200];
    assert_eq!(unsafe { mid_net_quic_recv_datagram(p.served, buf.as_mut_ptr(), buf.len()) }, 0, "it was dropped, not kept");
    close_pair(p);
}

#[test]
fn reliable_messages_arrive_whole_in_order_with_their_stream_ids() {
    let p = connected_pair();
    for i in 0..20u8 {
        let msg = vec![i; 1 + usize::from(i) * 7];
        assert_eq!(unsafe { mid_net_quic_send_reliable(p.client, 3, msg.as_ptr(), msg.len()) }, 0);
    }
    for i in 0..20u8 {
        let (id, msg) = recv_rel(p.served, 4096);
        assert_eq!(id, 3);
        assert_eq!(msg, vec![i; 1 + usize::from(i) * 7], "message {i} came out wrong or out of order");
    }
    let other = b"on another stream";
    assert_eq!(unsafe { mid_net_quic_send_reliable(p.served, 9, other.as_ptr(), other.len()) }, 0);
    assert_eq!(recv_rel(p.client, 64), (9, other.to_vec()));
    close_pair(p);
}

#[test]
fn a_big_reliable_message_arrives_in_one_piece() {
    let p = connected_pair();
    let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    assert_eq!(unsafe { mid_net_quic_send_reliable(p.client, 1, big.as_ptr(), big.len()) }, 0);
    let (id, got) = recv_rel(p.served, 300_000);
    assert_eq!(id, 1);
    assert_eq!(got, big);
    close_pair(p);
}

#[test]
fn an_empty_reliable_message_is_refused_and_a_small_buffer_gets_a_code() {
    let p = connected_pair();
    let x = [1u8];
    assert_eq!(unsafe { mid_net_quic_send_reliable(p.client, 0, x.as_ptr(), 0) }, MidNetQuicStatus::SendFailed as i32);
    let msg = vec![9u8; 50];
    assert_eq!(unsafe { mid_net_quic_send_reliable(p.client, 0, msg.as_ptr(), msg.len()) }, 0);
    let mut tiny = [0u8; 5];
    let mut got = 0;
    wait_for("the message", 5000, || {
        got = unsafe { mid_net_quic_recv_reliable(p.served, ptr::null_mut(), tiny.as_mut_ptr(), tiny.len()) };
        got != 0
    });
    assert_eq!(got, MidNetQuicStatus::BufferTooSmall as i32);
    close_pair(p);
}

#[test]
fn a_message_sent_just_before_closing_with_a_linger_still_arrives() {
    let p = connected_pair();
    let last = b"goodbye";
    assert_eq!(unsafe { mid_net_quic_send_reliable(p.client, 0, last.as_ptr(), last.len()) }, 0);
    unsafe { mid_net_quic_conn_close(p.client, 500) };
    assert_eq!(recv_rel(p.served, 64), (0, last.to_vec()));
    unsafe {
        mid_net_quic_conn_close(p.served, 0);
        mid_net_quic_server_close(p.server);
    }
}

#[test]
fn when_the_peer_closes_this_end_says_closed() {
    let p = connected_pair();
    unsafe { mid_net_quic_conn_close(p.client, 0) };
    wait_for("the server's end to notice", 5000, || unsafe { mid_net_quic_conn_state(p.served) } == MID_NET_QUIC_CLOSED);
    let x = [1u8];
    assert_eq!(unsafe { mid_net_quic_send_datagram(p.served, x.as_ptr(), 1) }, MidNetQuicStatus::Closed as i32);
    assert_eq!(unsafe { mid_net_quic_send_reliable(p.served, 0, x.as_ptr(), 1) }, MidNetQuicStatus::Closed as i32);
    let mut buf = [0u8; 8];
    assert_eq!(unsafe { mid_net_quic_recv_datagram(p.served, buf.as_mut_ptr(), 8) }, MidNetQuicStatus::Closed as i32);
    unsafe {
        mid_net_quic_conn_close(p.served, 0);
        mid_net_quic_server_close(p.server);
    }
}

#[test]
fn several_clients_each_get_their_own_connection() {
    let server = open_server();
    let port = unsafe { mid_net_quic_server_port(server) };
    let host = CString::new("127.0.0.1").unwrap();
    let mut clients = Vec::new();
    for _ in 0..3 {
        let mut c = ptr::null_mut();
        assert_eq!(unsafe { mid_net_quic_connect(host.as_ptr(), port, &mut c) }, 0);
        clients.push(c);
    }
    let mut served = Vec::new();
    wait_for("three accepts", 8000, || {
        let mut s = ptr::null_mut();
        if unsafe { mid_net_quic_server_accept(server, &mut s) } == 1 {
            served.push(s);
        }
        served.len() == 3
    });
    for (i, &c) in clients.iter().enumerate() {
        let msg = [i as u8; 4];
        wait_for("connected", 5000, || unsafe { mid_net_quic_conn_state(c) } == MID_NET_QUIC_CONNECTED);
        assert_eq!(unsafe { mid_net_quic_send_datagram(c, msg.as_ptr(), 4) }, 0);
    }
    // every payload shows up on exactly one server-side connection, and they are all different
    let mut seen = Vec::new();
    wait_for("three payloads", 5000, || {
        for &s in &served {
            let mut buf = [0u8; 16];
            let n = unsafe { mid_net_quic_recv_datagram(s, buf.as_mut_ptr(), 16) };
            if n == 4 {
                seen.push(buf[0]);
            }
        }
        seen.len() >= 3
    });
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2]);
    unsafe {
        for c in clients {
            mid_net_quic_conn_close(c, 0);
        }
        for s in served {
            mid_net_quic_conn_close(s, 0);
        }
        mid_net_quic_server_close(server);
    }
}

#[test]
fn bad_arguments_give_codes_not_crashes() {
    let mut conn: *mut MidNetQuicConn = ptr::null_mut();
    let mut server: *mut MidNetQuicServer = ptr::null_mut();
    let empty = CString::new("").unwrap();
    let junk = CString::new("not an address").unwrap();
    unsafe {
        assert_eq!(mid_net_quic_server_open(ptr::null(), 0, ptr::null_mut()), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_server_open(junk.as_ptr(), 0, &mut server), MidNetQuicStatus::InvalidAddress as i32);
        assert_eq!(mid_net_quic_connect(ptr::null(), 1, &mut conn), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_connect(empty.as_ptr(), 1, &mut conn), MidNetQuicStatus::InvalidAddress as i32);
        assert_eq!(mid_net_quic_connect(empty.as_ptr(), 1, ptr::null_mut()), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_server_port(ptr::null()), 0);
        assert_eq!(mid_net_quic_server_accept(ptr::null_mut(), &mut conn), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_conn_state(ptr::null_mut()), MidNetQuicStatus::NullPointer as i32);
        let mut buf = [0u8; 4];
        assert_eq!(mid_net_quic_send_datagram(ptr::null_mut(), buf.as_ptr(), 4), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_send_datagram(ptr::null_mut(), ptr::null(), 4), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_recv_datagram(ptr::null_mut(), buf.as_mut_ptr(), 4), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_recv_reliable(ptr::null_mut(), ptr::null_mut(), buf.as_mut_ptr(), 4), MidNetQuicStatus::NullPointer as i32);
        assert_eq!(mid_net_quic_conn_error(ptr::null_mut(), ptr::null_mut::<c_char>(), 0), MidNetQuicStatus::NullPointer as i32);
        mid_net_quic_conn_close(ptr::null_mut(), 0); // closing nothing is fine
        mid_net_quic_server_close(ptr::null_mut());
    }
}

#[test]
fn a_connection_still_connecting_says_so_instead_of_sending() {
    // Port 9 (discard) on loopback: nothing listens, so this stays connecting until the handshake times out.
    let host = CString::new("127.0.0.1").unwrap();
    let mut c = ptr::null_mut();
    assert_eq!(unsafe { mid_net_quic_connect(host.as_ptr(), 9, &mut c) }, 0);
    assert_eq!(unsafe { mid_net_quic_conn_state(c) }, MID_NET_QUIC_CONNECTING);
    let x = [1u8];
    assert_eq!(unsafe { mid_net_quic_send_datagram(c, x.as_ptr(), 1) }, MidNetQuicStatus::NotConnected as i32);
    unsafe { mid_net_quic_conn_close(c, 0) };
}

#[test]
fn a_closed_connection_explains_itself() {
    let p = connected_pair();
    unsafe { mid_net_quic_conn_close(p.client, 0) };
    wait_for("closed", 5000, || unsafe { mid_net_quic_conn_state(p.served) } == MID_NET_QUIC_CLOSED);
    let n = unsafe { mid_net_quic_conn_error(p.served, ptr::null_mut(), 0) };
    assert_eq!(n, 0, "a connection the peer closed has no failure text of its own");
    unsafe {
        mid_net_quic_conn_close(p.served, 0);
        mid_net_quic_server_close(p.server);
    }
}
