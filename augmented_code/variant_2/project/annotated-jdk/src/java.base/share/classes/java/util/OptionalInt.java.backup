/*
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.optional.qual.EnsuresPresent;
import org.checkerframework.checker.optional.qual.EnsuresPresentIf;
import org.checkerframework.checker.optional.qual.OptionalCreator;
import org.checkerframework.checker.optional.qual.OptionalEliminator;
import org.checkerframework.checker.optional.qual.Present;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.function.IntConsumer;
import java.util.function.IntSupplier;
import java.util.function.Supplier;
import java.util.stream.IntStream;

@AnnotatedFor({ "lock", "nullness", "optional" })
@jdk.internal.ValueBased
@NonNull
public final class OptionalInt {

    @OptionalCreator
    @SideEffectFree
    public static OptionalInt empty();

    @OptionalCreator
    @SideEffectFree
    @Present
    public static OptionalInt of(int value);

    @OptionalEliminator
    @Pure
    public int getAsInt(@Present OptionalInt this);

    @OptionalEliminator
    @Pure
    @EnsuresPresentIf(result = true, expression = "this")
    public boolean isPresent();

    @Pure
    @EnsuresPresentIf(result = false, expression = "this")
    public boolean isEmpty();

    @OptionalEliminator
    public void ifPresent(IntConsumer action);

    @OptionalEliminator
    public void ifPresentOrElse(IntConsumer action, Runnable emptyAction);

    @SideEffectFree
    public IntStream stream();

    @OptionalEliminator
    public int orElse(int other);

    @OptionalEliminator
    public int orElseGet(IntSupplier supplier);

    @OptionalEliminator
    @Pure
    @EnsuresPresent("this")
    public int orElseThrow(@Present OptionalInt this);

    @EnsuresPresent("this")
    @OptionalEliminator
    public <X extends Throwable> int orElseThrow(Supplier<? extends X> exceptionSupplier) throws X;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @OptionalEliminator
    @Pure
    @Override
    public int hashCode();

    @Override
    public String toString();
}
