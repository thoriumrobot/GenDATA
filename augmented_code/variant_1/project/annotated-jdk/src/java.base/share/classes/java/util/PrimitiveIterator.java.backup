/*
 * Copyright (c) 2013, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntConsumer;
import java.util.function.LongConsumer;

@AnnotatedFor({ "lock", "nullness" })
public interface PrimitiveIterator<T, T_CONS> extends Iterator<T> {

    @SuppressWarnings("overloads")
    void forEachRemaining(T_CONS action);

    public static interface OfInt extends PrimitiveIterator<Integer, IntConsumer> {

        int nextInt(@NonEmpty OfInt this);

        default void forEachRemaining(IntConsumer action);

        @Override
        default Integer next(PrimitiveIterator.@GuardSatisfied OfInt this);

        @Override
        default void forEachRemaining(Consumer<? super Integer> action);
    }

    public static interface OfLong extends PrimitiveIterator<Long, LongConsumer> {

        long nextLong(@NonEmpty OfLong this);

        default void forEachRemaining(LongConsumer action);

        @Override
        default Long next(PrimitiveIterator.@GuardSatisfied OfLong this);

        @Override
        default void forEachRemaining(Consumer<? super Long> action);
    }

    public static interface OfDouble extends PrimitiveIterator<Double, DoubleConsumer> {

        double nextDouble(@NonEmpty OfDouble this);

        default void forEachRemaining(DoubleConsumer action);

        @Override
        default Double next(PrimitiveIterator.@GuardSatisfied OfDouble this);

        @Override
        default void forEachRemaining(Consumer<? super Double> action);
    }
}
