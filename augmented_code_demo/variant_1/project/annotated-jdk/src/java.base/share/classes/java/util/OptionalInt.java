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
import org.checkerframework.checker.optional.qual.EnsuresPresent;
    @Positive
import org.checkerframework.checker.optional.qual.EnsuresPresentIf;
    @Positive
import org.checkerframework.checker.optional.qual.OptionalCreator;
    @Positive
import org.checkerframework.checker.optional.qual.OptionalEliminator;
    @Positive
import org.checkerframework.checker.optional.qual.Present;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.function.IntSupplier;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.IntStream;

    @Positive
@AnnotatedFor({ "lock", "nullness", "optional" })
    @Positive
@jdk.internal.ValueBased
    @Positive
@NonNull
    @Positive
public final class OptionalInt {

    @Positive
    @OptionalCreator
    @Positive
    @SideEffectFree
    @Positive
    public static OptionalInt empty();

    @Positive
    @OptionalCreator
    @Positive
    @SideEffectFree
    @Positive
    @Present
    @Positive
    public static OptionalInt of(int value);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    public int getAsInt(@Present OptionalInt this);

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
    public void ifPresent(IntConsumer action);

    @Positive
    @OptionalEliminator
    @Positive
    public void ifPresentOrElse(IntConsumer action, Runnable emptyAction);

    @Positive
    @SideEffectFree
    @Positive
    public IntStream stream();

    @Positive
    @OptionalEliminator
    @Positive
    public int orElse(int other);

    @Positive
    @OptionalEliminator
    @Positive
    public int orElseGet(IntSupplier supplier);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @EnsuresPresent("this")
    @Positive
    public int orElseThrow(@Present OptionalInt this);

    @Positive
    @EnsuresPresent("this")
    @Positive
    @OptionalEliminator
    @Positive
    public <X extends Throwable> int orElseThrow(Supplier<? extends X> exceptionSupplier) throws X;

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
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
