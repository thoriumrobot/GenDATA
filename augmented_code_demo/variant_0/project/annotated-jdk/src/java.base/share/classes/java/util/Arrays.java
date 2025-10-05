/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.SearchIndexFor;
    @Positive
import org.checkerframework.checker.interning.qual.PolyInterned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallUnknown;
    @Positive
import org.checkerframework.checker.mustcall.qual.PolyMustCall;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.checker.signedness.qual.Unsigned;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import jdk.internal.util.ArraysSupport;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.concurrent.ForkJoinPool;
    @Positive
import java.util.function.BinaryOperator;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.DoubleBinaryOperator;
    @Positive
import java.util.function.IntBinaryOperator;
    @Positive
import java.util.function.IntFunction;
    @Positive
import java.util.function.IntToDoubleFunction;
    @Positive
import java.util.function.IntToLongFunction;
    @Positive
import java.util.function.IntUnaryOperator;
    @Positive
import java.util.function.LongBinaryOperator;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import java.util.stream.DoubleStream;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.LongStream;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness", "signedness" })
    @Positive
public class Arrays {

    @Positive
    public static void sort(int[] a);

    @Positive
    public static void sort(int[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void sort(long[] a);

    @Positive
    public static void sort(long[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void sort(short[] a);

    @Positive
    public static void sort(short[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void sort(char[] a);

    @Positive
    public static void sort(char[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void sort(byte[] a);

    @Positive
    public static void sort(byte[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void sort(float[] a);

    @Positive
    public static void sort(float[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void sort(double[] a);

    @Positive
    public static void sort(double[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(byte[] a);

    @Positive
    public static void parallelSort(byte[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(char[] a);

    @Positive
    public static void parallelSort(char[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(short[] a);

    @Positive
    public static void parallelSort(short[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(int[] a);

    @Positive
    public static void parallelSort(int[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(long[] a);

    @Positive
    public static void parallelSort(long[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(float[] a);

    @Positive
    public static void parallelSort(float[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static void parallelSort(double[] a);

    @Positive
    public static void parallelSort(double[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    static void rangeCheck(int arrayLength, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    static final class NaturalOrder implements Comparator<Object> {

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public int compare(Object first, Object second);
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T extends Comparable<? super T>> void parallelSort(T[] a);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T extends Comparable<? super T>> void parallelSort(T[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> void parallelSort(T[] a, @Nullable Comparator<? super T> cmp);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> void parallelSort(T[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, Comparator<? super T> cmp);

    @Positive
    static final class LegacyMergeSort {
    @Positive
    }

    @Positive
    public static void sort(@PolyInterned @PolyNull Object[] a);

    @Positive
    public static void sort(@PolyInterned @PolyNull Object[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex);

    @Positive
    public static <T> void sort(@PolyNull @UnknownSignedness T[] a, @Nullable Comparator<? super T> c);

    @Positive
    public static <T> void sort(T[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, Comparator<? super T> c);

    @Positive
    public static <T> void parallelPrefix(T[] array, BinaryOperator<T> op);

    @Positive
    public static <T> void parallelPrefix(T[] array, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, BinaryOperator<T> op);

    @Positive
    public static void parallelPrefix(long[] array, LongBinaryOperator op);

    @Positive
    public static void parallelPrefix(long[] array, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, LongBinaryOperator op);

    @Positive
    public static void parallelPrefix(double[] array, DoubleBinaryOperator op);

    @Positive
    public static void parallelPrefix(double[] array, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, DoubleBinaryOperator op);

    @Positive
    public static void parallelPrefix(int[] array, IntBinaryOperator op);

    @Positive
    public static void parallelPrefix(int[] array, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, IntBinaryOperator op);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(long[] a, long key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(long[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, long key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(int[] a, int key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(int[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, int key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(short[] a, short key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(short[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, short key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(char[] a, char key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(char[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, char key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(byte[] a, byte key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(byte[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, byte key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(double[] a, double key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(double[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, double key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(float[] a, float key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(float[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, float key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(@Nullable @PolyInterned Object[] a, @Nullable @PolyInterned Object key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static int binarySearch(@Nullable @PolyInterned Object[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @Nullable @PolyInterned Object key);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static <T> int binarySearch(T[] a, T key, @Nullable Comparator<? super T> c);

    @Positive
    @SearchIndexFor({ "#1" })
    @Positive
    public static <T> int binarySearch(T[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, T key, @Nullable Comparator<? super T> c);

    @Positive
    @Pure
    @Positive
    public static boolean equals(@PolySigned long @Nullable [] a, @PolySigned long @Nullable [] a2);

    @Positive
    public static boolean equals(long[] a, int aFromIndex, int aToIndex, long[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    public static boolean equals(@PolySigned int @Nullable [] a, @PolySigned int @Nullable [] a2);

    @Positive
    public static boolean equals(int[] a, int aFromIndex, int aToIndex, int[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    public static boolean equals(@PolySigned short @Nullable [] a, @PolySigned short @Nullable [] a2);

    @Positive
    public static boolean equals(short[] a, int aFromIndex, int aToIndex, short[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static boolean equals(@PolySigned char @Nullable [] a, @PolySigned char @Nullable [] a2);

    @Positive
    public static boolean equals(char[] a, int aFromIndex, int aToIndex, char[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static boolean equals(@PolySigned byte @Nullable [] a, @PolySigned byte @Nullable [] a2);

    @Positive
    public static boolean equals(byte[] a, int aFromIndex, int aToIndex, byte[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    public static boolean equals(boolean @Nullable [] a, boolean @Nullable [] a2);

    @Positive
    public static boolean equals(boolean[] a, int aFromIndex, int aToIndex, boolean[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    public static boolean equals(double @Nullable [] a, double @Nullable [] a2);

    @Positive
    public static boolean equals(double[] a, int aFromIndex, int aToIndex, double[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    public static boolean equals(float @Nullable [] a, float @Nullable [] a2);

    @Positive
    public static boolean equals(float[] a, int aFromIndex, int aToIndex, float[] b, int bFromIndex, int bToIndex);

    @Positive
    @Pure
    @Positive
    public static boolean equals(@PolyInterned @PolyNull @PolySigned Object @GuardSatisfied @Nullable [] a, @PolyInterned @PolyNull @PolySigned Object @GuardSatisfied @Nullable [] a2);

    @Positive
    public static boolean equals(Object[] a, int aFromIndex, int aToIndex, Object[] b, int bFromIndex, int bToIndex);

    @Positive
    public static <T> boolean equals(T[] a, T[] a2, Comparator<? super T> cmp);

    @Positive
    public static <T> boolean equals(T[] a, int aFromIndex, int aToIndex, T[] b, int bFromIndex, int bToIndex, Comparator<? super T> cmp);

    @Positive
    public static void fill(@PolySigned long[] a, @PolySigned long val);

    @Positive
    public static void fill(@PolySigned long[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @PolySigned long val);

    @Positive
    public static void fill(@PolySigned int[] a, @PolySigned int val);

    @Positive
    public static void fill(@PolySigned int[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @PolySigned int val);

    @Positive
    public static void fill(@PolySigned short[] a, @PolySigned short val);

    @Positive
    public static void fill(@PolySigned short[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @PolySigned short val);

    @Positive
    public static void fill(@PolySigned char[] a, @PolySigned char val);

    @Positive
    public static void fill(@PolySigned char[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @PolySigned char val);

    @Positive
    public static void fill(@PolySigned byte[] a, @PolySigned byte val);

    @Positive
    public static void fill(@PolySigned byte[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @PolySigned byte val);

    @Positive
    public static void fill(boolean[] a, boolean val);

    @Positive
    public static void fill(boolean[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, boolean val);

    @Positive
    public static void fill(double[] a, double val);

    @Positive
    public static void fill(double[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, double val);

    @Positive
    public static void fill(float[] a, float val);

    @Positive
    public static void fill(float[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, float val);

    @Positive
    public static void fill(@PolyInterned @PolyNull @PolySigned Object[] a, @PolyInterned @PolyNull @PolySigned Object val);

    @Positive
    public static void fill(@PolyInterned @PolyNull @PolySigned Object[] a, @IndexOrHigh({ "#1" }) int fromIndex, @IndexOrHigh({ "#1" }) int toIndex, @PolyInterned @PolyNull @PolySigned Object val);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public static <T> T[] copyOf(T[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    @IntrinsicCandidate
    @Positive
    @Nullable
    @Positive
    public static <T, U> T[] copyOf(U[] original, @NonNegative int newLength, Class<? extends T[]> newType);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static byte[] copyOf(@PolySigned byte[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static short[] copyOf(@PolySigned short[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static int[] copyOf(@PolySigned int[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static long[] copyOf(@PolySigned long[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static char[] copyOf(@PolySigned char[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    public static float[] copyOf(float[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    public static double[] copyOf(double[] original, @NonNegative int newLength);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean[] copyOf(boolean[] original, @NonNegative int newLength);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public static <T> T[] copyOfRange(T[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @IntrinsicCandidate
    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public static <T, U> T[] copyOfRange(U[] original, @IndexOrHigh({ "#1" }) int from, int to, Class<? extends T[]> newType);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static byte[] copyOfRange(@PolySigned byte[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static short[] copyOfRange(@PolySigned short[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static int[] copyOfRange(@PolySigned int[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static long[] copyOfRange(@PolySigned long[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    @PolySigned
    @Positive
    public static char[] copyOfRange(@PolySigned char[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    public static float[] copyOfRange(float[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    public static double[] copyOfRange(double[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean[] copyOfRange(boolean[] original, @IndexOrHigh({ "#1" }) int from, int to);

    @Positive
    @SafeVarargs
    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("varargs")
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> List<T> asList(T@PolyNonEmpty ... a);

    @Positive
    private static class ArrayList<E> extends AbstractList<E> implements RandomAccess, java.io.Serializable {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Arrays.ArrayList<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public <T> T[] toArray(T[] a);

    @Positive
        @Override
    @Positive
        public E get(int index);

    @Positive
        @Override
    @Positive
        public E set(int index, E element);

    @Positive
        @Override
    @Positive
        public int indexOf(Object o);

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);

    @Positive
        @Override
    @Positive
        public Iterator<E> iterator();
    @Positive
    }

    @Positive
    private static class ArrayItr<E> implements Iterator<E> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty ArrayItr<E> this);
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public static int hashCode(@PolySigned long @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(@PolySigned int @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(@PolySigned short @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(@PolySigned char @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(@PolySigned byte @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(boolean @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(float @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(double @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int hashCode(@PolyInterned @PolyNull @PolySigned Object @GuardSatisfied @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static int deepHashCode(@PolyInterned @PolyNull @PolySigned Object @GuardSatisfied @Nullable [] a);

    @Positive
    @Pure
    @Positive
    public static boolean deepEquals(@PolyInterned @PolyNull @PolySigned Object @GuardSatisfied @Nullable [] a1, @PolyInterned @PolyNull @PolySigned Object @GuardSatisfied @Nullable [] a2);

    @Positive
    static boolean deepEquals0(Object e1, Object e2);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(long @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(int @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(short @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(char @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(byte @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(boolean @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(float @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(double @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @CFComment({ "The @PolyMustCall annotations don't make sense, because toString", "shouldn't care about MustCall types, especially of the array.  However,", "without these annotations, calls to Arrays.toString yield a MustCall error." })
    @Positive
    @MinLen(2)
    @Positive
    public static String toString(@PolyInterned @PolyMustCall @PolyNull @PolySigned Object @PolyMustCall @Nullable [] a);

    @Positive
    @SideEffectFree
    @Positive
    @MinLen(2)
    @Positive
    public static String deepToString(@PolyInterned @PolyMustCall @PolyNull @PolySigned Object @PolyMustCall @Nullable [] a);

    @Positive
    public static <T> void setAll(T[] array, IntFunction<? extends T> generator);

    @Positive
    public static <T> void parallelSetAll(T[] array, IntFunction<? extends T> generator);

    @Positive
    public static void setAll(int[] array, IntUnaryOperator generator);

    @Positive
    public static void parallelSetAll(int[] array, IntUnaryOperator generator);

    @Positive
    public static void setAll(long[] array, IntToLongFunction generator);

    @Positive
    public static void parallelSetAll(long[] array, IntToLongFunction generator);

    @Positive
    public static void setAll(double[] array, IntToDoubleFunction generator);

    @Positive
    public static void parallelSetAll(double[] array, IntToDoubleFunction generator);

    @Positive
    @SideEffectFree
    @Positive
    public static <T> Spliterator<T> spliterator(T[] array);

    @Positive
    @SideEffectFree
    @Positive
    public static <T> Spliterator<T> spliterator(T[] array, int startInclusive, int endExclusive);

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfInt spliterator(int[] array);

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfInt spliterator(int[] array, int startInclusive, int endExclusive);

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfLong spliterator(long[] array);

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfLong spliterator(long[] array, int startInclusive, int endExclusive);

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfDouble spliterator(double[] array);

    @Positive
    @SideEffectFree
    @Positive
    public static Spliterator.OfDouble spliterator(double[] array, int startInclusive, int endExclusive);

    @Positive
    public static <T> Stream<T> stream(T[] array);

    @Positive
    public static <T> Stream<T> stream(T[] array, int startInclusive, int endExclusive);

    @Positive
    public static IntStream stream(int[] array);

    @Positive
    public static IntStream stream(int[] array, int startInclusive, int endExclusive);

    @Positive
    public static LongStream stream(long[] array);

    @Positive
    public static LongStream stream(long[] array, int startInclusive, int endExclusive);

    @Positive
    public static DoubleStream stream(double[] array);

    @Positive
    public static DoubleStream stream(double[] array, int startInclusive, int endExclusive);

    @Positive
    public static int compare(boolean[] a, boolean[] b);

    @Positive
    public static int compare(boolean[] a, int aFromIndex, int aToIndex, boolean[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compare(byte[] a, byte[] b);

    @Positive
    public static int compare(byte[] a, int aFromIndex, int aToIndex, byte[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compareUnsigned(@Unsigned byte[] a, @Unsigned byte[] b);

    @Positive
    public static int compareUnsigned(@Unsigned byte[] a, @IndexFor("#1") int aFromIndex, @IndexFor("#1") int aToIndex, @Unsigned byte[] b, @IndexFor("#3") int bFromIndex, @IndexFor("#3") int bToIndex);

    @Positive
    public static int compare(short[] a, short[] b);

    @Positive
    public static int compare(short[] a, int aFromIndex, int aToIndex, short[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compareUnsigned(@Unsigned short[] a, @Unsigned short[] b);

    @Positive
    public static int compareUnsigned(@Unsigned short[] a, @IndexFor("#1") int aFromIndex, @IndexFor("#1") int aToIndex, @Unsigned short[] b, @IndexFor("#3") int bFromIndex, @IndexFor("#3") int bToIndex);

    @Positive
    public static int compare(char[] a, char[] b);

    @Positive
    public static int compare(char[] a, int aFromIndex, int aToIndex, char[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compare(int[] a, int[] b);

    @Positive
    public static int compare(int[] a, int aFromIndex, int aToIndex, int[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compareUnsigned(@Unsigned int[] a, @Unsigned int[] b);

    @Positive
    public static int compareUnsigned(@Unsigned int[] a, @IndexFor("#1") int aFromIndex, @IndexFor("#1") int aToIndex, @Unsigned int[] b, @IndexFor("#3") int bFromIndex, @IndexFor("#3") int bToIndex);

    @Positive
    public static int compare(long[] a, long[] b);

    @Positive
    public static int compare(long[] a, int aFromIndex, int aToIndex, long[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compareUnsigned(@Unsigned long[] a, @Unsigned long[] b);

    @Positive
    public static int compareUnsigned(@Unsigned long[] a, @IndexFor("#1") int aFromIndex, @IndexFor("#1") int aToIndex, @Unsigned long[] b, @IndexFor("#3") int bFromIndex, @IndexFor("#3") int bToIndex);

    @Positive
    public static int compare(float[] a, float[] b);

    @Positive
    public static int compare(float[] a, int aFromIndex, int aToIndex, float[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int compare(double[] a, double[] b);

    @Positive
    public static int compare(double[] a, int aFromIndex, int aToIndex, double[] b, int bFromIndex, int bToIndex);

    @Positive
    public static <T extends Comparable<? super T>> int compare(T[] a, T[] b);

    @Positive
    public static <T extends Comparable<? super T>> int compare(T[] a, int aFromIndex, int aToIndex, T[] b, int bFromIndex, int bToIndex);

    @Positive
    public static <T> int compare(T[] a, T[] b, Comparator<? super T> cmp);

    @Positive
    public static <T> int compare(T[] a, int aFromIndex, int aToIndex, T[] b, int bFromIndex, int bToIndex, Comparator<? super T> cmp);

    @Positive
    public static int mismatch(boolean[] a, boolean[] b);

    @Positive
    public static int mismatch(boolean[] a, int aFromIndex, int aToIndex, boolean[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(byte[] a, byte[] b);

    @Positive
    public static int mismatch(byte[] a, int aFromIndex, int aToIndex, byte[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(char[] a, char[] b);

    @Positive
    public static int mismatch(char[] a, int aFromIndex, int aToIndex, char[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(short[] a, short[] b);

    @Positive
    public static int mismatch(short[] a, int aFromIndex, int aToIndex, short[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(int[] a, int[] b);

    @Positive
    public static int mismatch(int[] a, int aFromIndex, int aToIndex, int[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(long[] a, long[] b);

    @Positive
    public static int mismatch(long[] a, int aFromIndex, int aToIndex, long[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(float[] a, float[] b);

    @Positive
    public static int mismatch(float[] a, int aFromIndex, int aToIndex, float[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(double[] a, double[] b);

    @Positive
    public static int mismatch(double[] a, int aFromIndex, int aToIndex, double[] b, int bFromIndex, int bToIndex);

    @Positive
    public static int mismatch(Object[] a, Object[] b);

    @Positive
    public static int mismatch(Object[] a, int aFromIndex, int aToIndex, Object[] b, int bFromIndex, int bToIndex);

    @Positive
    public static <T> int mismatch(T[] a, T[] b, Comparator<? super T> cmp);

    @Positive
    public static <T> int mismatch(T[] a, int aFromIndex, int aToIndex, T[] b, int bFromIndex, int bToIndex, Comparator<? super T> cmp);
    @Positive
}

// CFWR semantic augmentation - variant 0
