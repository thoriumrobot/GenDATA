/*
 * Copyright (c) 2009, 2019, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.concurrent.CountedCompleter;
import java.util.concurrent.RecursiveTask;

@AnnotatedFor({ "index", "interning" })
@UsesObjectEquals
final class DualPivotQuicksort {

    static void sort(int[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(Sorter sorter, int[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(long[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(Sorter sorter, long[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(byte[] a, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(char[] a, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(char[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(short[] a, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(short[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(float[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(Sorter sorter, float[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(double[] a, int parallelism, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    static void sort(Sorter sorter, double[] a, int bits, @IndexOrHigh({ "#1" }) int low, @IndexOrHigh({ "#1" }) int high);

    private static final class Sorter extends CountedCompleter<Void> {

        @Override
        public final void compute();

        @Override
        public final void onCompletion(CountedCompleter<?> caller);
    }

    private static final class Merger extends CountedCompleter<Void> {

        @Override
        public final void compute();
    }

    private static final class RunMerger extends RecursiveTask<Object> {

        @Override
        protected final Object compute();
    }
}
