/*
    @Positive
 * Copyright (c) 2013, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.DoubleConsumer;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.function.LongConsumer;

    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface Spliterator<T> {

    @Positive
    boolean tryAdvance(Consumer<? super T> action);

    @Positive
    default void forEachRemaining(Consumer<? super T> action);

    @Positive
    @Nullable
    @Positive
    Spliterator<T> trySplit();

    @Positive
    long estimateSize();

    @Positive
    default long getExactSizeIfKnown();

    @Positive
    int characteristics();

    @Positive
    default boolean hasCharacteristics(int characteristics);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    default Comparator<? super T> getComparator();

    @Positive
    @SignedPositive
    @Positive
    public static final int ORDERED;

    @Positive
    @SignedPositive
    @Positive
    public static final int DISTINCT;

    @Positive
    @SignedPositive
    @Positive
    public static final int SORTED;

    @Positive
    @SignedPositive
    @Positive
    public static final int SIZED;

    @Positive
    @SignedPositive
    @Positive
    public static final int NONNULL;

    @Positive
    @SignedPositive
    @Positive
    public static final int IMMUTABLE;

    @Positive
    @SignedPositive
    @Positive
    public static final int CONCURRENT;

    @Positive
    @SignedPositive
    @Positive
    public static final int SUBSIZED;

    @Positive
    public interface OfPrimitive<T, T_CONS, T_SPLITR extends Spliterator.OfPrimitive<T, T_CONS, T_SPLITR>> extends Spliterator<T> {

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        T_SPLITR trySplit();

    @Positive
        @SuppressWarnings("overloads")
    @Positive
        boolean tryAdvance(T_CONS action);

    @Positive
        @SuppressWarnings("overloads")
    @Positive
        default void forEachRemaining(T_CONS action);
    @Positive
    }

    @Positive
    public interface OfInt extends OfPrimitive<Integer, IntConsumer, OfInt> {

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        OfInt trySplit();

    @Positive
        @Override
    @Positive
        boolean tryAdvance(IntConsumer action);

    @Positive
        @Override
    @Positive
        default void forEachRemaining(IntConsumer action);

    @Positive
        @Override
    @Positive
        default boolean tryAdvance(Consumer<? super Integer> action);

    @Positive
        @Override
    @Positive
        default void forEachRemaining(Consumer<? super Integer> action);
    @Positive
    }

    @Positive
    public interface OfLong extends OfPrimitive<Long, LongConsumer, OfLong> {

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        OfLong trySplit();

    @Positive
        @Override
    @Positive
        boolean tryAdvance(LongConsumer action);

    @Positive
        @Override
    @Positive
        default void forEachRemaining(LongConsumer action);

    @Positive
        @Override
    @Positive
        default boolean tryAdvance(Consumer<? super Long> action);

    @Positive
        @Override
    @Positive
        default void forEachRemaining(Consumer<? super Long> action);
    @Positive
    }

    @Positive
    public interface OfDouble extends OfPrimitive<Double, DoubleConsumer, OfDouble> {

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        OfDouble trySplit();

    @Positive
        @Override
    @Positive
        boolean tryAdvance(DoubleConsumer action);

    @Positive
        @Override
    @Positive
        default void forEachRemaining(DoubleConsumer action);

    @Positive
        @Override
    @Positive
        default boolean tryAdvance(Consumer<? super Double> action);

    @Positive
        @Override
    @Positive
        default void forEachRemaining(Consumer<? super Double> action);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
