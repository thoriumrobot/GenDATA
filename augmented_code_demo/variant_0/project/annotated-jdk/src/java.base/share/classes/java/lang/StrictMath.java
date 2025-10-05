/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Random;
    @Positive
import jdk.internal.math.DoubleConsts;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class StrictMath {

    @Positive
    public static final double E;

    @Positive
    public static final double PI;

    @Positive
    public static native double sin(double a);

    @Positive
    public static native double cos(double a);

    @Positive
    public static native double tan(double a);

    @Positive
    public static native double asin(double a);

    @Positive
    public static native double acos(double a);

    @Positive
    public static native double atan(double a);

    @Positive
    public static double toRadians(double angdeg);

    @Positive
    public static double toDegrees(double angrad);

    @Positive
    public static double exp(double a);

    @Positive
    public static native double log(double a);

    @Positive
    public static native double log10(double a);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static native double sqrt(double a);

    @Positive
    public static double cbrt(double a);

    @Positive
    public static native double IEEEremainder(double f1, double f2);

    @Positive
    public static double ceil(double a);

    @Positive
    public static double floor(double a);

    @Positive
    public static double rint(double a);

    @Positive
    public static native double atan2(double y, double x);

    @Positive
    public static double pow(double a, double b);

    @Positive
    public static int round(float a);

    @Positive
    public static long round(double a);

    @Positive
    private static final class RandomNumberGeneratorHolder {
    @Positive
    }

    @Positive
    public static double random();

    @Positive
    public static int addExact(int x, int y);

    @Positive
    public static long addExact(long x, long y);

    @Positive
    public static int subtractExact(int x, int y);

    @Positive
    public static long subtractExact(long x, long y);

    @Positive
    public static int multiplyExact(int x, int y);

    @Positive
    public static long multiplyExact(long x, int y);

    @Positive
    public static long multiplyExact(long x, long y);

    @Positive
    @Pure
    @Positive
    public static int incrementExact(int a);

    @Positive
    @Pure
    @Positive
    public static long incrementExact(long a);

    @Positive
    @Pure
    @Positive
    public static int decrementExact(int a);

    @Positive
    @Pure
    @Positive
    public static long decrementExact(long a);

    @Positive
    @Pure
    @Positive
    public static int negateExact(int a);

    @Positive
    @Pure
    @Positive
    public static long negateExact(long a);

    @Positive
    public static int toIntExact(long value);

    @Positive
    public static long multiplyFull(int x, int y);

    @Positive
    public static long multiplyHigh(long x, long y);

    @Positive
    public static int floorDiv(int x, int y);

    @Positive
    public static long floorDiv(long x, int y);

    @Positive
    public static long floorDiv(long x, long y);

    @Positive
    public static int floorMod(int x, int y);

    @Positive
    public static int floorMod(long x, int y);

    @Positive
    public static long floorMod(long x, long y);

    @Positive
    public static int abs(int a);

    @Positive
    @Pure
    @Positive
    public static int absExact(int a);

    @Positive
    public static long abs(long a);

    @Positive
    @Pure
    @Positive
    public static long absExact(long a);

    @Positive
    public static float abs(float a);

    @Positive
    public static double abs(double a);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static int max(int a, int b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long max(long a, long b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float max(float a, float b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double max(double a, double b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static int min(int a, int b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static long min(long a, long b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float min(float a, float b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double min(double a, double b);

    @Positive
    public static double fma(double a, double b, double c);

    @Positive
    public static float fma(float a, float b, float c);

    @Positive
    public static double ulp(double d);

    @Positive
    public static float ulp(float f);

    @Positive
    public static double signum(double d);

    @Positive
    public static float signum(float f);

    @Positive
    public static native double sinh(double x);

    @Positive
    public static native double cosh(double x);

    @Positive
    public static native double tanh(double x);

    @Positive
    public static double hypot(double x, double y);

    @Positive
    public static native double expm1(double x);

    @Positive
    public static native double log1p(double x);

    @Positive
    public static double copySign(double magnitude, double sign);

    @Positive
    public static float copySign(float magnitude, float sign);

    @Positive
    public static int getExponent(float f);

    @Positive
    public static int getExponent(double d);

    @Positive
    public static double nextAfter(double start, double direction);

    @Positive
    public static float nextAfter(float start, double direction);

    @Positive
    public static double nextUp(double d);

    @Positive
    public static float nextUp(float f);

    @Positive
    public static double nextDown(double d);

    @Positive
    public static float nextDown(float f);

    @Positive
    public static double scalb(double d, int scaleFactor);

    @Positive
    public static float scalb(float f, int scaleFactor);
    @Positive
}

// CFWR semantic augmentation - variant 0
