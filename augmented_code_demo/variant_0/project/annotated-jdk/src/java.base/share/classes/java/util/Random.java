/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import java.util.random.RandomGenerator;
    @Positive
import java.util.stream.DoubleStream;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.LongStream;
    @Positive
import jdk.internal.util.random.RandomSupport.*;
    @Positive
import static jdk.internal.util.random.RandomSupport.*;
    @Positive
import jdk.internal.misc.Unsafe;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness", "signedness" })
    @Positive
@SuppressWarnings("exports")
    @Positive
@RandomGeneratorProperties(name = "Random", i = 48, j = 0, k = 0, equidistribution = 0)
    @Positive
@UsesObjectEquals
    @Positive
public class Random implements RandomGenerator, java.io.Serializable {

    @Positive
    public Random() {
    @Positive
    }

    @Positive
    public Random(long seed) {
    @Positive
    }

    @Positive
    public synchronized void setSeed(@GuardSatisfied Random this, long seed);

    @Positive
    protected int next(int bits);

    @Positive
    @Override
    @Positive
    public void nextBytes(@PolySigned byte[] bytes);

    @Positive
    @Override
    @Positive
    public int nextInt();

    @Positive
    @Override
    @Positive
    @NonNegative
    @Positive
    public int nextInt(@Positive int bound);

    @Positive
    @Override
    @Positive
    public long nextLong();

    @Positive
    @Override
    @Positive
    public boolean nextBoolean();

    @Positive
    @Override
    @Positive
    public float nextFloat();

    @Positive
    @Override
    @Positive
    public double nextDouble();

    @Positive
    @Override
    @Positive
    public synchronized double nextGaussian();

    @Positive
    @Override
    @Positive
    public IntStream ints(long streamSize);

    @Positive
    @Override
    @Positive
    public IntStream ints();

    @Positive
    @Override
    @Positive
    public IntStream ints(long streamSize, int randomNumberOrigin, int randomNumberBound);

    @Positive
    @Override
    @Positive
    public IntStream ints(int randomNumberOrigin, int randomNumberBound);

    @Positive
    @Override
    @Positive
    public LongStream longs(long streamSize);

    @Positive
    @Override
    @Positive
    public LongStream longs();

    @Positive
    @Override
    @Positive
    public LongStream longs(long streamSize, long randomNumberOrigin, long randomNumberBound);

    @Positive
    @Override
    @Positive
    public LongStream longs(long randomNumberOrigin, long randomNumberBound);

    @Positive
    @Override
    @Positive
    public DoubleStream doubles(long streamSize);

    @Positive
    @Override
    @Positive
    public DoubleStream doubles();

    @Positive
    @Override
    @Positive
    public DoubleStream doubles(long streamSize, double randomNumberOrigin, double randomNumberBound);

    @Positive
    @Override
    @Positive
    public DoubleStream doubles(double randomNumberOrigin, double randomNumberBound);
    @Positive
}

// CFWR semantic augmentation - variant 0
