/*
    @Positive
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.util;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.optional.qual.EnsuresPresent;
    @Positive
import org.checkerframework.checker.optional.qual.EnsuresPresentIf;
    @Positive
import org.checkerframework.checker.optional.qual.OptionalCreator;
    @Positive
import org.checkerframework.checker.optional.qual.OptionalEliminator;
    @Positive
import org.checkerframework.checker.optional.qual.OptionalPropagator;
    @Positive
import org.checkerframework.checker.optional.qual.Present;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import org.checkerframework.framework.qual.Covariant;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.Stream;

    @Positive
@CFComment({ "nullness :", "The @NonNull annotation on the class makes the type \"@Nullable Optional<T>\" illegal and enforces", "\"Rule #1: Never, ever, use null for an Optional variable or return value.\" from", "https://stuartmarks.files.wordpress.com/2016/09/optionalmotherofallbikesheds3.pdf, which is", "generally accepted practice.  If you wish to permit the type \"@Nullable Optional\", you may do so", "by writing a stub file that overrides this class in the annotated JDK.", "The type argument to Optional is meaningless.", "Optional<@NonNull String> and Optional<@Nullable String> have the same", "meaning, but are unrelated by the Java type hierarchy.", "@Covariant makes Optional<@NonNull String> a subtype of Optional<@Nullable String>." })
    @Positive
@AnnotatedFor({ "lock", "nullness", "optional" })
    @Positive
@Covariant(0)
    @Positive
@jdk.internal.ValueBased
    @Positive
@NonNull
    @Positive
public final class Optional<T> {

    @Positive
    @OptionalCreator
    @Positive
    @Pure
    @Positive
    public static <T> Optional<T> empty();

    @Positive
    @OptionalCreator
    @Positive
    @SideEffectFree
    @Positive
    @Present
    @Positive
    public static <T> Optional<T> of(@NonNull T value);

    @Positive
    @OptionalCreator
    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> Optional<@NonNull T> ofNullable(@Nullable T value);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @NonNull
    @Positive
    public T get(@Present Optional<T> this);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @EnsuresPresentIf(result = true, expression = "this")
    @Positive
    public boolean isPresent();

    @Positive
    @Pure
    @Positive
    @EnsuresPresentIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    @OptionalEliminator
    @Positive
    public void ifPresent(Consumer<? super T> action);

    @Positive
    @OptionalEliminator
    @Positive
    public void ifPresentOrElse(Consumer<? super T> action, Runnable emptyAction);

    @Positive
    @OptionalPropagator
    @Positive
    public Optional<T> filter(Predicate<? super T> predicate);

    @Positive
    @CFComment({ "@SideEffectFree: the mapper must not have side effects." })
    @Positive
    @OptionalPropagator
    @Positive
    @SideEffectFree
    @Positive
    public <U> Optional<U> map(Function<? super T, ? extends @Nullable U> mapper);

    @Positive
    @OptionalPropagator
    @Positive
    public <U> Optional<U> flatMap(Function<? super T, ? extends Optional<? extends U>> mapper);

    @Positive
    @OptionalPropagator
    @Positive
    public Optional<T> or(Supplier<? extends Optional<? extends T>> supplier);

    @Positive
    @SideEffectFree
    @Positive
    public Stream<T> stream();

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @PolyNull
    @Positive
    public T orElse(@PolyNull T other);

    @Positive
    @OptionalEliminator
    @Positive
    @PolyNull
    @Positive
    public T orElseGet(Supplier<? extends @PolyNull T> supplier);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @EnsuresPresent("this")
    @Positive
    public T orElseThrow(@Present Optional<T> this);

    @Positive
    @CFComment({ "optional: orElseThrow(Supplier) does not throw NoSuchElementException, so its receiver is @MaybePresent.", "Contrast with orElseThrow(), defined just above, whose receiver is @Present." })
    @Positive
    @EnsuresPresent("this")
    @Positive
    @OptionalEliminator
    @Positive
    public <X extends Throwable> T orElseThrow(Supplier<? extends X> exceptionSupplier) throws X;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
