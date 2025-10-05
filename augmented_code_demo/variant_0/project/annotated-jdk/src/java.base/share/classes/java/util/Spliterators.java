/*
    @Positive
 * Copyright (c) 2013, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.DoubleConsumer;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.function.LongConsumer;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;

    @Positive
public final class Spliterators {

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static <T> Spliterator<T> emptySpliterator();

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfInt emptyIntSpliterator();

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfLong emptyLongSpliterator();

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfDouble emptyDoubleSpliterator();

    @Positive
    public static <T> Spliterator<T> spliterator(Object[] array, int additionalCharacteristics);

    @Positive
    public static <T> Spliterator<T> spliterator(Object[] array, int fromIndex, int toIndex, int additionalCharacteristics);

    @Positive
    public static Spliterator.OfInt spliterator(int[] array, int additionalCharacteristics);

    @Positive
    public static Spliterator.OfInt spliterator(int[] array, int fromIndex, int toIndex, int additionalCharacteristics);

    @Positive
    public static Spliterator.OfLong spliterator(long[] array, int additionalCharacteristics);

    @Positive
    public static Spliterator.OfLong spliterator(long[] array, int fromIndex, int toIndex, int additionalCharacteristics);

    @Positive
    public static Spliterator.OfDouble spliterator(double[] array, int additionalCharacteristics);

    @Positive
    public static Spliterator.OfDouble spliterator(double[] array, int fromIndex, int toIndex, int additionalCharacteristics);

    @Positive
    public static <T> Spliterator<T> spliterator(Collection<? extends T> c, int characteristics);

    @Positive
    public static <T> Spliterator<T> spliterator(Iterator<? extends T> iterator, long size, int characteristics);

    @Positive
    public static <T> Spliterator<T> spliteratorUnknownSize(Iterator<? extends T> iterator, int characteristics);

    @Positive
    public static Spliterator.OfInt spliterator(PrimitiveIterator.OfInt iterator, long size, int characteristics);

    @Positive
    public static Spliterator.OfInt spliteratorUnknownSize(PrimitiveIterator.OfInt iterator, int characteristics);

    @Positive
    public static Spliterator.OfLong spliterator(PrimitiveIterator.OfLong iterator, long size, int characteristics);

    @Positive
    public static Spliterator.OfLong spliteratorUnknownSize(PrimitiveIterator.OfLong iterator, int characteristics);

    @Positive
    public static Spliterator.OfDouble spliterator(PrimitiveIterator.OfDouble iterator, long size, int characteristics);

    @Positive
    public static Spliterator.OfDouble spliteratorUnknownSize(PrimitiveIterator.OfDouble iterator, int characteristics);

    @Positive
    public static <T> Iterator<T> iterator(Spliterator<? extends T> spliterator);

    @Positive
    public static PrimitiveIterator.OfInt iterator(Spliterator.OfInt spliterator);

    @Positive
    public static PrimitiveIterator.OfLong iterator(Spliterator.OfLong spliterator);

    @Positive
    public static PrimitiveIterator.OfDouble iterator(Spliterator.OfDouble spliterator);

    @Positive
    private abstract static class EmptySpliterator<T, S extends Spliterator<T>, C> {

    @Positive
        public S trySplit();

    @Positive
        public boolean tryAdvance(C consumer);

    @Positive
        public void forEachRemaining(C consumer);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();

    @Positive
        private static final class OfRef<T> extends EmptySpliterator<T, Spliterator<T>, Consumer<? super T>> implements Spliterator<T> {
    @Positive
        }

    @Positive
        private static final class OfInt extends EmptySpliterator<Integer, Spliterator.OfInt, IntConsumer> implements Spliterator.OfInt {
    @Positive
        }

    @Positive
        private static final class OfLong extends EmptySpliterator<Long, Spliterator.OfLong, LongConsumer> implements Spliterator.OfLong {
    @Positive
        }

    @Positive
        private static final class OfDouble extends EmptySpliterator<Double, Spliterator.OfDouble, DoubleConsumer> implements Spliterator.OfDouble {
    @Positive
        }
    @Positive
    }

    @Positive
    static final class ArraySpliterator<T> implements Spliterator<T> {

    @Positive
        public ArraySpliterator(Object[] array, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        public ArraySpliterator(Object[] array, int origin, int fence, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Spliterator<T> trySplit();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super T> action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(Consumer<? super T> action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super T> getComparator();
    @Positive
    }

    @Positive
    static final class IntArraySpliterator implements Spliterator.OfInt {

    @Positive
        public IntArraySpliterator(int[] array, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        public IntArraySpliterator(int[] array, int origin, int fence, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public OfInt trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(IntConsumer action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(IntConsumer action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super Integer> getComparator();
    @Positive
    }

    @Positive
    static final class LongArraySpliterator implements Spliterator.OfLong {

    @Positive
        public LongArraySpliterator(long[] array, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        public LongArraySpliterator(long[] array, int origin, int fence, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public OfLong trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(LongConsumer action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(LongConsumer action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super Long> getComparator();
    @Positive
    }

    @Positive
    static final class DoubleArraySpliterator implements Spliterator.OfDouble {

    @Positive
        public DoubleArraySpliterator(double[] array, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        public DoubleArraySpliterator(double[] array, int origin, int fence, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public OfDouble trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(DoubleConsumer action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(DoubleConsumer action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super Double> getComparator();
    @Positive
    }

    @Positive
    public abstract static class AbstractSpliterator<T> implements Spliterator<T> {

    @Positive
        protected AbstractSpliterator(long est, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        static final class HoldingConsumer<T> implements Consumer<T> {

    @Positive
            @Override
    @Positive
            public void accept(T value);
    @Positive
        }

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        public Spliterator<T> trySplit();

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public abstract static class AbstractIntSpliterator implements Spliterator.OfInt {

    @Positive
        protected AbstractIntSpliterator(long est, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        static final class HoldingIntConsumer implements IntConsumer {

    @Positive
            @Override
    @Positive
            public void accept(int value);
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Spliterator.@Nullable OfInt trySplit();

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public abstract static class AbstractLongSpliterator implements Spliterator.OfLong {

    @Positive
        protected AbstractLongSpliterator(long est, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        static final class HoldingLongConsumer implements LongConsumer {

    @Positive
            @Override
    @Positive
            public void accept(long value);
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Spliterator.@Nullable OfLong trySplit();

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public abstract static class AbstractDoubleSpliterator implements Spliterator.OfDouble {

    @Positive
        protected AbstractDoubleSpliterator(long est, int additionalCharacteristics) {
    @Positive
        }

    @Positive
        static final class HoldingDoubleConsumer implements DoubleConsumer {

    @Positive
            @Override
    @Positive
            public void accept(double value);
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Spliterator.@Nullable OfDouble trySplit();

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    static class IteratorSpliterator<T> implements Spliterator<T> {

    @Positive
        public IteratorSpliterator(Collection<? extends T> collection, int characteristics) {
    @Positive
        }

    @Positive
        public IteratorSpliterator(Iterator<? extends T> iterator, long size, int characteristics) {
    @Positive
        }

    @Positive
        public IteratorSpliterator(Iterator<? extends T> iterator, int characteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Spliterator<T> trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super T> action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(Consumer<? super T> action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super T> getComparator();
    @Positive
    }

    @Positive
    static final class IntIteratorSpliterator implements Spliterator.OfInt {

    @Positive
        public IntIteratorSpliterator(PrimitiveIterator.OfInt iterator, long size, int characteristics) {
    @Positive
        }

    @Positive
        public IntIteratorSpliterator(PrimitiveIterator.OfInt iterator, int characteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public OfInt trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(IntConsumer action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(IntConsumer action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super Integer> getComparator();
    @Positive
    }

    @Positive
    static final class LongIteratorSpliterator implements Spliterator.OfLong {

    @Positive
        public LongIteratorSpliterator(PrimitiveIterator.OfLong iterator, long size, int characteristics) {
    @Positive
        }

    @Positive
        public LongIteratorSpliterator(PrimitiveIterator.OfLong iterator, int characteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public OfLong trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(LongConsumer action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(LongConsumer action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super Long> getComparator();
    @Positive
    }

    @Positive
    static final class DoubleIteratorSpliterator implements Spliterator.OfDouble {

    @Positive
        public DoubleIteratorSpliterator(PrimitiveIterator.OfDouble iterator, long size, int characteristics) {
    @Positive
        }

    @Positive
        public DoubleIteratorSpliterator(PrimitiveIterator.OfDouble iterator, int characteristics) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public OfDouble trySplit();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(DoubleConsumer action);

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(DoubleConsumer action);

    @Positive
        @Override
    @Positive
        public long estimateSize();

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<? super Double> getComparator();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
