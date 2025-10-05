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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntConsumer;
import java.util.function.LongConsumer;

@AnnotatedFor({ "lock", "nullness" })
public interface Spliterator<T> {

    boolean tryAdvance(Consumer<? super T> action);

    default void forEachRemaining(Consumer<? super T> action);

    @Nullable
    Spliterator<T> trySplit();

    long estimateSize();

    default long getExactSizeIfKnown();

    int characteristics();

    default boolean hasCharacteristics(int characteristics);

    @Pure
    @Nullable
    default Comparator<? super T> getComparator();

    @SignedPositive
    public static final int ORDERED;

    @SignedPositive
    public static final int DISTINCT;

    @SignedPositive
    public static final int SORTED;

    @SignedPositive
    public static final int SIZED;

    @SignedPositive
    public static final int NONNULL;

    @SignedPositive
    public static final int IMMUTABLE;

    @SignedPositive
    public static final int CONCURRENT;

    @SignedPositive
    public static final int SUBSIZED;

    public interface OfPrimitive<T, T_CONS, T_SPLITR extends Spliterator.OfPrimitive<T, T_CONS, T_SPLITR>> extends Spliterator<T> {

        @Override
        @Nullable
        T_SPLITR trySplit();

        @SuppressWarnings("overloads")
        boolean tryAdvance(T_CONS action);

        @SuppressWarnings("overloads")
        default void forEachRemaining(T_CONS action);
    }

    public interface OfInt extends OfPrimitive<Integer, IntConsumer, OfInt> {

        @Override
        @Nullable
        OfInt trySplit();

        @Override
        boolean tryAdvance(IntConsumer action);

        @Override
        default void forEachRemaining(IntConsumer action);

        @Override
        default boolean tryAdvance(Consumer<? super Integer> action);

        @Override
        default void forEachRemaining(Consumer<? super Integer> action);
    }

    public interface OfLong extends OfPrimitive<Long, LongConsumer, OfLong> {

        @Override
        @Nullable
        OfLong trySplit();

        @Override
        boolean tryAdvance(LongConsumer action);

        @Override
        default void forEachRemaining(LongConsumer action);

        @Override
        default boolean tryAdvance(Consumer<? super Long> action);

        @Override
        default void forEachRemaining(Consumer<? super Long> action);
    }

    public interface OfDouble extends OfPrimitive<Double, DoubleConsumer, OfDouble> {

        @Override
        @Nullable
        OfDouble trySplit();

        @Override
        boolean tryAdvance(DoubleConsumer action);

        @Override
        default void forEachRemaining(DoubleConsumer action);

        @Override
        default boolean tryAdvance(Consumer<? super Double> action);

        @Override
        default void forEachRemaining(Consumer<? super Double> action);
    }
}
