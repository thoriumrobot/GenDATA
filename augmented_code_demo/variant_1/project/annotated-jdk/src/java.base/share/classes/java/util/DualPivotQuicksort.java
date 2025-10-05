/*
    @Positive
 * Copyright (c) 2009, 2019, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.concurrent.CountedCompleter;
    @Positive
import java.util.concurrent.RecursiveTask;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
final class DualPivotQuicksort {

    @Positive
    static void sort(int[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(Sorter sorter, int[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(long[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(Sorter sorter, long[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(byte[] a, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(char[] a, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(char[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(short[] a, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(short[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(float[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(Sorter sorter, float[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(double[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    static void sort(Sorter sorter, double[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    @Positive
    private static final class Sorter extends CountedCompleter<Void> {

    @Positive
        @Override
    @Positive
        public final void compute();

    @Positive
        @Override
    @Positive
        public final void onCompletion(CountedCompleter<?> caller);
    @Positive
    }

    @Positive
    private static final class Merger extends CountedCompleter<Void> {

    @Positive
        @Override
    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    private static final class RunMerger extends RecursiveTask<Object> {

    @Positive
        @Override
    @Positive
        protected final Object compute();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
