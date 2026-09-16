// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "moving_macros.rs"
// ============================================================================
//
// Needs Rust 1.82+: the field-projection arms below use the `&raw mut`/`&raw const`
// raw-reference operators (stabilized in 1.82, RFC 2582), not available on this
// workspace's usual rustc-1.75 floor. See the root Cargo.toml's MSRV-wall comment
// block and docs/mid-ptr.md for the full note.

/// Safely converts an owned value into a [`crate::MovingPtr`] while minimizing the
/// number of stack copies.
///
/// This cannot be used as an expression — only as a statement. Internally it works
/// by shadowing the original binding with a `MaybeUninit`, then with the
/// `MovingPtr` built from it, so there's no way left to name the original value.
#[macro_export]
macro_rules! move_as_ptr {
    ($value: ident) => {
        let mut $value = ::core::mem::MaybeUninit::new($value);
        // SAFETY:
        // - This macro shadows a `MaybeUninit` value that took ownership of the
        //   original value; it's impossible to refer to the original after this,
        //   preventing further access once the `MovingPtr` has been used.
        //   `MaybeUninit` also stops the compiler from dropping the original value
        //   on its own.
        let $value = unsafe { $crate::MovingPtr::from_value(&mut $value) };
    };
}

/// Helper macro used by [`deconstruct_moving_ptr`] to extract the pattern from a
/// `field: pattern` or bare `field` shorthand.
#[macro_export]
#[doc(hidden)]
macro_rules! get_pattern {
    ($field_index:tt) => {
        $field_index
    };
    ($field_index:tt: $pattern:pat) => {
        $pattern
    };
}

/// Deconstructs a [`crate::MovingPtr`] into its individual fields.
///
/// Consumes the `MovingPtr` and hands out `MovingPtr` wrappers around pointers to
/// each of its fields. The value itself is *not* dropped.
///
/// The macro wraps a `let` expression with a struct pattern. It does not support
/// matching tuples by position, so tuple structs need `0: pat` syntax.
///
/// For tuples themselves, pass the identifier `tuple` instead of the struct name,
/// like `let tuple { 0: pat0, 1: pat1 } = value`.
///
/// This can also project through `MaybeUninit`: wrap the type name (or `tuple`)
/// with `MaybeUninit::<_>`, and the macro deconstructs a
/// `MovingPtr<MaybeUninit<ParentType>>` into `MovingPtr<MaybeUninit<FieldType>>`
/// values.
///
/// # Examples
///
/// ## Structs
///
/// ```
/// use core::mem::{offset_of, MaybeUninit};
/// use mid_ptr::{MovingPtr, move_as_ptr};
/// # use mid_ptr::Unaligned;
/// # struct FieldAType(usize);
/// # struct FieldBType(usize);
/// # struct FieldCType(usize);
///
/// # pub struct Parent {
/// #  pub field_a: FieldAType,
/// #  pub field_b: FieldBType,
/// #  pub field_c: FieldCType,
/// # }
///
/// let parent = Parent {
///   field_a: FieldAType(11),
///   field_b: FieldBType(22),
///   field_c: FieldCType(33),
/// };
///
/// let mut target_a = FieldAType(101);
/// let mut target_b = FieldBType(102);
/// let mut target_c = FieldCType(103);
///
/// // Converts `parent` into a `MovingPtr`.
/// move_as_ptr!(parent);
///
/// // The field names must match the ones used in the type definition. Each one
/// // ends up a `MovingPtr` of the field's own type.
/// mid_ptr::deconstruct_moving_ptr!({
///   let Parent { field_a, field_b, field_c } = parent;
/// });
///
/// field_a.assign_to(&mut target_a);
/// field_b.assign_to(&mut target_b);
/// field_c.assign_to(&mut target_c);
///
/// assert_eq!(target_a.0, 11);
/// assert_eq!(target_b.0, 22);
/// assert_eq!(target_c.0, 33);
/// ```
///
/// ## Tuples
///
/// ```
/// use core::mem::{offset_of, MaybeUninit};
/// use mid_ptr::{MovingPtr, move_as_ptr};
/// # use mid_ptr::Unaligned;
/// # struct FieldAType(usize);
/// # struct FieldBType(usize);
/// # struct FieldCType(usize);
///
/// let parent = (
///   FieldAType(11),
///   FieldBType(22),
///   FieldCType(33),
/// );
///
/// let mut target_a = FieldAType(101);
/// let mut target_b = FieldBType(102);
/// let mut target_c = FieldCType(103);
///
/// // Converts `parent` into a `MovingPtr`.
/// move_as_ptr!(parent);
///
/// // The field names must match the name used in the type definition.
/// // Each one will be a `MovingPtr` of the field's type.
/// mid_ptr::deconstruct_moving_ptr!({
///   let tuple { 0: field_a, 1: field_b, 2: field_c } = parent;
/// });
///
/// field_a.assign_to(&mut target_a);
/// field_b.assign_to(&mut target_b);
/// field_c.assign_to(&mut target_c);
///
/// assert_eq!(target_a.0, 11);
/// assert_eq!(target_b.0, 22);
/// assert_eq!(target_c.0, 33);
/// ```
///
/// ## `MaybeUninit`
///
/// ```
/// use core::mem::{offset_of, MaybeUninit};
/// use mid_ptr::{MovingPtr, move_as_ptr};
/// # use mid_ptr::Unaligned;
/// # struct FieldAType(usize);
/// # struct FieldBType(usize);
/// # struct FieldCType(usize);
///
/// # pub struct Parent {
/// #  pub field_a: FieldAType,
/// #  pub field_b: FieldBType,
/// #  pub field_c: FieldCType,
/// # }
///
/// let parent = MaybeUninit::new(Parent {
///   field_a: FieldAType(11),
///   field_b: FieldBType(22),
///   field_c: FieldCType(33),
/// });
///
/// let mut target_a = MaybeUninit::new(FieldAType(101));
/// let mut target_b = MaybeUninit::new(FieldBType(102));
/// let mut target_c = MaybeUninit::new(FieldCType(103));
///
/// // Converts `parent` into a `MovingPtr`.
/// move_as_ptr!(parent);
///
/// // The field names must match the name used in the type definition.
/// // Each one will be a `MovingPtr` of the field's type.
/// mid_ptr::deconstruct_moving_ptr!({
///   let MaybeUninit::<Parent> { field_a, field_b, field_c } = parent;
/// });
///
/// field_a.assign_to(&mut target_a);
/// field_b.assign_to(&mut target_b);
/// field_c.assign_to(&mut target_c);
///
/// unsafe {
///   assert_eq!(target_a.assume_init().0, 11);
///   assert_eq!(target_b.assume_init().0, 22);
///   assert_eq!(target_c.assume_init().0, 33);
/// }
/// ```
///
/// [`assign_to`]: crate::MovingPtr::assign_to
#[macro_export]
macro_rules! deconstruct_moving_ptr {
    ({ let tuple { $($field_index:tt: $pattern:pat),* $(,)? } = $ptr:expr ;}) => {
        // Specify the type so `mem::forget` below can't accidentally forget a mere
        // `&mut MovingPtr`.
        let mut ptr: $crate::MovingPtr<_, _> = $ptr;
        let _ = || {
            let value = &mut *ptr;
            // Ensure every field index exists and is mentioned only once.
            // Ensure the struct is not `repr(packed)` and that taking references to
            // its fields is sound.
            ::core::hint::black_box(($(&mut value.$field_index,)*));
            // Ensure `ptr` is a tuple and not merely something that derefs to one.
            // Ensure the pattern count matches the field count.
            fn unreachable<T>(_index: usize) -> T {
                ::core::unreachable!()
            }
            *value = ($(unreachable($field_index),)*);
        };
        // SAFETY:
        // - `f` does a raw pointer offset, which always returns a non-null pointer
        //   to a field inside `T`.
        // - The struct is not `repr(packed)`, since otherwise the block above would
        //   fail to compile.
        // - `mem::forget` is called on `self` immediately after these calls.
        // - Each field is distinct, since otherwise the block above would fail to
        //   compile.
        $(let $pattern = unsafe { ptr.move_field(|f| &raw mut (*f).$field_index) };)*
        #[expect(clippy::mem_forget, reason = "`deconstruct_moving_ptr` needs to forget the `MovingPtr` due to its safety requirements.")]
        ::core::mem::forget(ptr);
    };
    ({ let MaybeUninit::<tuple> { $($field_index:tt: $pattern:pat),* $(,)? } = $ptr:expr ;}) => {
        // Specify the type so `mem::forget` below can't accidentally forget a mere
        // `&mut MovingPtr`.
        let mut ptr: $crate::MovingPtr<::core::mem::MaybeUninit<_>, _> = $ptr;
        let _ = || {
            // SAFETY: This closure is never called.
            let value = unsafe { ptr.assume_init_mut() };
            // Ensure every field index exists and is mentioned only once.
            // Ensure the struct is not `repr(packed)` and that taking references to
            // its fields is sound.
            ::core::hint::black_box(($(&mut value.$field_index,)*));
            // Ensure `ptr` is a tuple and not merely something that derefs to one.
            // Ensure the pattern count matches the field count.
            fn unreachable<T>(_index: usize) -> T {
                ::core::unreachable!()
            }
            *value = ($(unreachable($field_index),)*);
        };
        // SAFETY:
        // - `f` does a raw pointer offset, which always returns a non-null pointer
        //   to a field inside `T`.
        // - The struct is not `repr(packed)`, since otherwise the block above would
        //   fail to compile.
        // - `mem::forget` is called on `self` immediately after these calls.
        // - Each field is distinct, since otherwise the block above would fail to
        //   compile.
        $(let $pattern = unsafe { ptr.move_maybe_uninit_field(|f| &raw mut (*f).$field_index) };)*
        #[expect(clippy::mem_forget, reason = "`deconstruct_moving_ptr` needs to forget the `MovingPtr` due to its safety requirements.")]
        ::core::mem::forget(ptr);
    };
    ({ let $struct_name:ident { $($field_index:tt$(: $pattern:pat)?),* $(,)? } = $ptr:expr ;}) => {
        // Specify the type so `mem::forget` below can't accidentally forget a mere
        // `&mut MovingPtr`.
        let mut ptr: $crate::MovingPtr<_, _> = $ptr;
        let _ = || {
            let value = &mut *ptr;
            // Ensure every field index exists and is mentioned only once.
            // Ensure each field is on the struct itself, not reached via autoref.
            let $struct_name { $($field_index: _),* } = value;
            // Ensure the struct is not `repr(packed)` and that taking references to
            // its fields is sound.
            ::core::hint::black_box(($(&mut value.$field_index),*));
            // Ensure `ptr` is really a `$struct_name` and not just something that
            // derefs to it.
            let value: *mut _ = value;
            // SAFETY: This closure is never called.
            $struct_name { ..unsafe { value.read() } };
        };
        // SAFETY:
        // - `f` does a raw pointer offset, which always returns a non-null pointer
        //   to a field inside `T`.
        // - The struct is not `repr(packed)`, since otherwise the block above would
        //   fail to compile.
        // - `mem::forget` is called on `self` immediately after these calls.
        // - Each field is distinct, since otherwise the block above would fail to
        //   compile.
        $(let $crate::get_pattern!($field_index$(: $pattern)?) = unsafe { ptr.move_field(|f| &raw mut (*f).$field_index) };)*
        #[expect(clippy::mem_forget, reason = "`deconstruct_moving_ptr` needs to forget the `MovingPtr` due to its safety requirements.")]
        ::core::mem::forget(ptr);
    };
    ({ let MaybeUninit::<$struct_name:ident> { $($field_index:tt$(: $pattern:pat)?),* $(,)? } = $ptr:expr ;}) => {
        // Specify the type so `mem::forget` below can't accidentally forget a mere
        // `&mut MovingPtr`.
        let mut ptr: $crate::MovingPtr<::core::mem::MaybeUninit<_>, _> = $ptr;
        let _ = || {
            // SAFETY: This closure is never called.
            let value = unsafe { ptr.assume_init_mut() };
            // Ensure every field index exists and is mentioned only once.
            // Ensure each field is on the struct itself, not reached via autoref.
            let $struct_name { $($field_index: _),* } = value;
            // Ensure the struct is not `repr(packed)` and that taking references to
            // its fields is sound.
            ::core::hint::black_box(($(&mut value.$field_index),*));
            // Ensure `ptr` is really a `$struct_name` and not just something that
            // derefs to it.
            let value: *mut _ = value;
            // SAFETY: This closure is never called.
            $struct_name { ..unsafe { value.read() } };
        };
        // SAFETY:
        // - `f` does a raw pointer offset, which always returns a non-null pointer
        //   to a field inside `T`.
        // - The struct is not `repr(packed)`, since otherwise the block above would
        //   fail to compile.
        // - `mem::forget` is called on `self` immediately after these calls.
        // - Each field is distinct, since otherwise the block above would fail to
        //   compile.
        $(let $crate::get_pattern!($field_index$(: $pattern)?) = unsafe { ptr.move_maybe_uninit_field(|f| &raw mut (*f).$field_index) };)*
        #[expect(clippy::mem_forget, reason = "`deconstruct_moving_ptr` needs to forget the `MovingPtr` due to its safety requirements.")]
        ::core::mem::forget(ptr);
    };
}
