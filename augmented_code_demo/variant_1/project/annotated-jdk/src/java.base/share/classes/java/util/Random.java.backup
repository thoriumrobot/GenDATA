/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import jdk.internal.util.random.RandomSupport.*;
import static jdk.internal.util.random.RandomSupport.*;
import jdk.internal.misc.Unsafe;

@AnnotatedFor({ "index", "interning", "lock", "nullness", "signedness" })
@SuppressWarnings("exports")
@RandomGeneratorProperties(name = "Random", i = 48, j = 0, k = 0, equidistribution = 0)
@UsesObjectEquals
public class Random implements RandomGenerator, java.io.Serializable {

    public Random() {
    }

    public Random(long seed) {
    }

    public synchronized void setSeed(@GuardSatisfied Random this, long seed);

    protected int next(int bits);

    @Override
    public void nextBytes(@PolySigned byte[] bytes);

    @Override
    public int nextInt();

    @Override
    @NonNegative
    public int nextInt(@Positive int bound);

    @Override
    public long nextLong();

    @Override
    public boolean nextBoolean();

    @Override
    public float nextFloat();

    @Override
    public double nextDouble();

    @Override
    public synchronized double nextGaussian();

    @Override
    public IntStream ints(long streamSize);

    @Override
    public IntStream ints();

    @Override
    public IntStream ints(long streamSize, int randomNumberOrigin, int randomNumberBound);

    @Override
    public IntStream ints(int randomNumberOrigin, int randomNumberBound);

    @Override
    public LongStream longs(long streamSize);

    @Override
    public LongStream longs();

    @Override
    public LongStream longs(long streamSize, long randomNumberOrigin, long randomNumberBound);

    @Override
    public LongStream longs(long randomNumberOrigin, long randomNumberBound);

    @Override
    public DoubleStream doubles(long streamSize);

    @Override
    public DoubleStream doubles();

    @Override
    public DoubleStream doubles(long streamSize, double randomNumberOrigin, double randomNumberBound);

    @Override
    public DoubleStream doubles(double randomNumberOrigin, double randomNumberBound);
}
