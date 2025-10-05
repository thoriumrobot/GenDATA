/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.util.function.DoubleConsumer;
    @Positive
import java.util.function.DoubleSupplier;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.DoubleStream;

    @Positive
@AnnotatedFor({ "lock", "nullness", "optional" })
    @Positive
@jdk.internal.ValueBased
    @Positive
@NonNull
    @Positive
public final class OptionalDouble {

    @Positive
    @OptionalCreator
    @Positive
    @SideEffectFree
    @Positive
    public static OptionalDouble empty();

    @Positive
    @OptionalCreator
    @Positive
    @SideEffectFree
    @Positive
    @Present
    @Positive
    public static OptionalDouble of(double value);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    public double getAsDouble(@Present OptionalDouble this);

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
    public void ifPresent(DoubleConsumer action);

    @Positive
    @OptionalEliminator
    @Positive
    public void ifPresentOrElse(DoubleConsumer action, Runnable emptyAction);

    @Positive
    @SideEffectFree
    @Positive
    public DoubleStream stream();

    @Positive
    @OptionalEliminator
    @Positive
    public double orElse(double other);

    @Positive
    @OptionalEliminator
    @Positive
    public double orElseGet(DoubleSupplier supplier);

    @Positive
    @OptionalEliminator
    @Positive
    @Pure
    @Positive
    @EnsuresPresent("this")
    @Positive
    public double orElseThrow(@Present OptionalDouble this);

    @Positive
    @OptionalEliminator
    @Positive
    @EnsuresPresent("this")
    @Positive
    public <X extends Throwable> double orElseThrow(Supplier<? extends X> exceptionSupplier) throws X;

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
