//! The same driver, run on a thread of its own, for a program that would rather not call `poll`.
//!
//! [`NetThread::spawn`] builds the [`Driver`] **on the new thread** (the closure you give it runs there), so
//! neither the driver nor your handler needs to be `Send`: the transport may hold a socket, the handler may
//! hold anything. The thread polls every `tick`, hands each received message to your handler (on the net
//! thread), and passes the driver's events back to you. What crosses threads are:
//!
//! * messages you **send** (they must be `Send`: plain data always is), through [`NetThread::send`];
//! * [`DriverEvent`]s, read with [`NetThread::next_event`];
//! * whatever your **handler** chooses to forward (a channel, a queue, a mutex): it runs on the net
//!   thread, and `mid-net` does not guess how your game wants to receive.
//!
//! The clock is milliseconds since the thread started, so build the driver with `now_ms` 0. A message that
//! the driver refuses (the connection has not finished, the channel is wrong, ...) comes back as
//! [`DriverEvent::SendRejected`], since by then `send` has already returned. The thread ends when the
//! connection ends, when [`NetThread::stop`] is called, or when the `NetThread` is dropped (which says goodbye
//! first); whichever it is, everything the driver had to report is delivered before it exits.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mid_net_transport::Transport;
use mid_net_wire::{NetMessage, NetMessageType, RawMessage};

use crate::driver::{Driver, DriverEvent, Inbound};
use crate::session::DisconnectReason;

enum Cmd {
    Send {
        channel: u8,
        priority: u8,
        msg: Box<dyn NetMessage + Send>,
    },
    Close(DisconnectReason),
    Stop,
}

/// The net thread has ended: nothing sent to it will be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadGone;

/// A [`Driver`] running on its own thread. See the module docs.
pub struct NetThread {
    cmds: Sender<Cmd>,
    events: Receiver<DriverEvent>,
    handle: Option<JoinHandle<()>>,
}

impl NetThread {
    /// Starts the thread. `make` runs on it and returns the driver and the message handler.
    pub fn spawn<T, H, F>(tick: Duration, make: F) -> NetThread
    where
        T: Transport + 'static,
        H: FnMut(Inbound) + 'static,
        F: FnOnce() -> (Driver<T>, H) + Send + 'static,
    {
        let (cmd_tx, cmd_rx) = channel::<Cmd>();
        let (ev_tx, ev_rx) = channel::<DriverEvent>();
        let handle = thread::Builder::new()
            .name("mid-net".to_string())
            .spawn(move || {
                let (driver, handler) = make();
                run(driver, handler, tick, cmd_rx, ev_tx);
            })
            .expect("the operating system refused to start a thread");
        NetThread {
            cmds: cmd_tx,
            events: ev_rx,
            handle: Some(handle),
        }
    }

    /// Queues a message for the driver. `Ok` means the thread will see it, not that the driver accepts it.
    pub fn send<M: NetMessageType + Send>(
        &self,
        channel: u8,
        priority: u8,
        msg: M,
    ) -> Result<(), ThreadGone> {
        self.send_boxed(channel, priority, Box::new(msg))
    }

    /// Opaque bytes under a message id registered with `register_raw`.
    pub fn send_raw(
        &self,
        channel: u8,
        priority: u8,
        id: u16,
        data: Vec<u8>,
    ) -> Result<(), ThreadGone> {
        self.send_boxed(channel, priority, Box::new(RawMessage { id, data }))
    }

    pub fn send_boxed(
        &self,
        channel: u8,
        priority: u8,
        msg: Box<dyn NetMessage + Send>,
    ) -> Result<(), ThreadGone> {
        self.cmds
            .send(Cmd::Send {
                channel,
                priority,
                msg,
            })
            .map_err(|_| ThreadGone)
    }

    /// Ends the connection on purpose (the goodbye is sent); the thread then finishes.
    pub fn close(&self, reason: DisconnectReason) -> Result<(), ThreadGone> {
        self.cmds.send(Cmd::Close(reason)).map_err(|_| ThreadGone)
    }

    /// The next event the driver reported, oldest first.
    pub fn next_event(&self) -> Option<DriverEvent> {
        self.events.try_recv().ok()
    }

    /// Waits up to `timeout` for the next event.
    pub fn wait_event(&self, timeout: Duration) -> Option<DriverEvent> {
        self.events.recv_timeout(timeout).ok()
    }

    /// Whether the thread is still running (it ends with the connection).
    pub fn is_running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Says goodbye, waits for the thread to finish, and returns the events not yet read.
    pub fn stop(mut self) -> Vec<DriverEvent> {
        self.shut_down();
        let mut rest = Vec::new();
        while let Some(e) = self.next_event() {
            rest.push(e);
        }
        rest
    }

    fn shut_down(&mut self) {
        let _ = self.cmds.send(Cmd::Stop);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for NetThread {
    fn drop(&mut self) {
        self.shut_down();
    }
}

fn run<T: Transport, H: FnMut(Inbound)>(
    mut driver: Driver<T>,
    mut handler: H,
    tick: Duration,
    cmds: Receiver<Cmd>,
    events: Sender<DriverEvent>,
) {
    let epoch = Instant::now();
    let now = |epoch: &Instant| epoch.elapsed().as_millis() as u64;
    loop {
        let mut stop = false;
        loop {
            match cmds.try_recv() {
                Ok(Cmd::Send {
                    channel,
                    priority,
                    msg,
                }) => {
                    let id = msg.id();
                    // dropping the `Send` marker: the driver takes any message
                    let msg: Box<dyn NetMessage> = msg;
                    if let Err(error) = driver.send_boxed(channel, priority, msg) {
                        let _ = events.send(DriverEvent::SendRejected {
                            channel,
                            id,
                            error,
                        });
                    }
                }
                Ok(Cmd::Close(reason)) => driver.close(reason),
                Ok(Cmd::Stop) | Err(TryRecvError::Disconnected) => {
                    stop = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if stop {
            driver.close(DisconnectReason::Normal);
        }
        driver.poll(now(&epoch));
        deliver(&mut driver, &mut handler, &events);
        if stop || driver.is_ended() {
            // one more pass so that the goodbye is on the wire and every event has been reported
            driver.poll(now(&epoch));
            deliver(&mut driver, &mut handler, &events);
            break;
        }
        thread::sleep(tick);
    }
}

fn deliver<T: Transport, H: FnMut(Inbound)>(
    driver: &mut Driver<T>,
    handler: &mut H,
    events: &Sender<DriverEvent>,
) {
    while let Some(m) = driver.next_message() {
        handler(m);
    }
    while let Some(e) = driver.next_event() {
        let _ = events.send(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{
        priority, DriverConfig, Schema, SendError, CONTROL_DATAGRAM_CHANNEL, CONTROL_RELIABLE_CHANNEL,
    };
    use crate::session::{EndReason, SessionEvent};
    use crate::sim::{SimConfig, SimNet, SimTransport};
    use mid_net_wire::{impl_net_serialize, ChannelKind, MessageRegistry};
    use std::sync::{Arc, Mutex};

    const CH_EVENT: u8 = 2;
    const CH_STATE: u8 = 3;

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Ev {
        n: u32,
    }
    impl_net_serialize!(Ev { n: raw() });
    impl NetMessageType for Ev {
        const ID: u16 = 5;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Other {
        a: u8,
    }
    impl_net_serialize!(Other { a: raw() });
    impl NetMessageType for Other {
        const ID: u16 = 6;
    }

    fn schema() -> Arc<Schema> {
        let mut reg = MessageRegistry::new(8);
        reg.register::<Ev>().unwrap();
        let mut ch = Schema::default_channels(4).unwrap();
        ch.define(CH_EVENT, ChannelKind::ReliableOrdered).unwrap();
        ch.define(CH_STATE, ChannelKind::UnreliableSequenced).unwrap();
        Arc::new(Schema::new(reg, ch, CONTROL_RELIABLE_CHANNEL, CONTROL_DATAGRAM_CHANNEL).unwrap())
    }

    type Log = Arc<Mutex<Vec<(u8, u32)>>>;

    fn spawn_end(client: bool, t: SimTransport, log: Log) -> NetThread {
        let sch = schema();
        NetThread::spawn(Duration::from_millis(1), move || {
            let cfg = DriverConfig::default();
            let d = if client {
                Driver::client(sch, cfg, t, "threaded", 9, 0).unwrap()
            } else {
                Driver::server_end(sch, cfg, t, 3, 0).unwrap()
            };
            let h = move |m: Inbound| {
                if let Ok(e) = m.downcast::<Ev>() {
                    log.lock().unwrap().push((0, e.n));
                }
            };
            (d, h)
        })
    }

    fn wait_for(t: &NetThread, what: impl Fn(&DriverEvent) -> bool) -> Vec<DriverEvent> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = Vec::new();
        while Instant::now() < deadline {
            if let Some(e) = t.wait_event(Duration::from_millis(20)) {
                let hit = what(&e);
                seen.push(e);
                if hit {
                    return seen;
                }
            }
        }
        panic!("the event never came: {seen:?}");
    }

    fn is_connected(e: &DriverEvent) -> bool {
        matches!(e, DriverEvent::Session(SessionEvent::Connected { .. }))
    }

    fn wait_until(what: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if what() {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out");
    }

    #[test]
    fn two_threads_connect_and_exchange_messages_in_order() {
        let (_net, a, b) = SimNet::pair(SimConfig::default(), 1);
        let (log_c, log_s): (Log, Log) = (Arc::default(), Arc::default());
        let client = spawn_end(true, a, Arc::clone(&log_c));
        let server = spawn_end(false, b, Arc::clone(&log_s));
        wait_for(&client, is_connected);
        wait_for(&server, is_connected);
        for n in 0..500u32 {
            client.send(CH_EVENT, priority::NORMAL, Ev { n }).unwrap();
        }
        server.send(CH_EVENT, priority::NORMAL, Ev { n: 7 }).unwrap();
        wait_until(|| log_s.lock().unwrap().len() == 500 && log_c.lock().unwrap().len() == 1);
        let got: Vec<u32> = log_s.lock().unwrap().iter().map(|(_, n)| *n).collect();
        assert_eq!(got, (0..500).collect::<Vec<_>>());
        assert_eq!(log_c.lock().unwrap()[0].1, 7);
        assert!(client.is_running() && server.is_running());
    }

    #[test]
    fn a_message_the_driver_refuses_comes_back_as_an_event() {
        let (_net, a, b) = SimNet::pair(SimConfig::default(), 1);
        let client = spawn_end(true, a, Arc::default());
        let _server = spawn_end(false, b, Arc::default());
        wait_for(&client, is_connected);
        client.send(CH_EVENT, 0, Other { a: 1 }).unwrap(); // not in the schema
        let seen = wait_for(&client, |e| matches!(e, DriverEvent::SendRejected { .. }));
        assert_eq!(
            seen.last(),
            Some(&DriverEvent::SendRejected {
                channel: CH_EVENT,
                id: 6,
                error: SendError::Unregistered
            })
        );
    }

    #[test]
    fn closing_says_goodbye_and_the_thread_finishes() {
        let (_net, a, b) = SimNet::pair(SimConfig::default(), 1);
        let client = spawn_end(true, a, Arc::default());
        let server = spawn_end(false, b, Arc::default());
        wait_for(&client, is_connected);
        wait_for(&server, is_connected);
        client.close(DisconnectReason::Normal).unwrap();
        wait_for(&client, |e| matches!(e, DriverEvent::Session(SessionEvent::Ended(EndReason::LocalClose))));
        wait_for(&server, |e| {
            matches!(e, DriverEvent::Session(SessionEvent::Ended(EndReason::PeerClosed(DisconnectReason::Normal))))
        });
        wait_until(|| !client.is_running() && !server.is_running());
        assert_eq!(client.send(CH_EVENT, 0, Ev { n: 1 }), Err(ThreadGone));
        assert_eq!(client.close(DisconnectReason::Normal), Err(ThreadGone));
    }

    #[test]
    fn dropping_a_net_thread_stops_it_and_the_peer_is_told() {
        let (_net, a, b) = SimNet::pair(SimConfig::default(), 1);
        let client = spawn_end(true, a, Arc::default());
        let server = spawn_end(false, b, Arc::default());
        wait_for(&client, is_connected);
        wait_for(&server, is_connected);
        drop(client);
        wait_for(&server, |e| {
            matches!(e, DriverEvent::Session(SessionEvent::Ended(EndReason::PeerClosed(DisconnectReason::Normal))))
        });
    }

    #[test]
    fn stop_returns_the_events_not_yet_read() {
        let (_net, a, b) = SimNet::pair(SimConfig::default(), 1);
        let client = spawn_end(true, a, Arc::default());
        let server = spawn_end(false, b, Arc::default());
        wait_for(&server, is_connected);
        let rest = client.stop();
        assert!(rest.iter().any(is_connected) || rest.iter().any(|e| matches!(e, DriverEvent::Session(SessionEvent::Ended(_)))), "{rest:?}");
        assert!(rest.iter().any(|e| matches!(e, DriverEvent::Session(SessionEvent::Ended(EndReason::LocalClose)))));
    }

    #[test]
    fn a_cut_link_ends_the_thread_with_a_fault() {
        let (net, a, b) = SimNet::pair(SimConfig::default(), 1);
        let client = spawn_end(true, a, Arc::default());
        let _server = spawn_end(false, b, Arc::default());
        wait_for(&client, is_connected);
        net.cut();
        wait_for(&client, |e| matches!(e, DriverEvent::Fault(_)));
        wait_until(|| !client.is_running());
    }
}
