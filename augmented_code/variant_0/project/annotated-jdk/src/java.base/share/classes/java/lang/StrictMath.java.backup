/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Random;
import jdk.internal.math.DoubleConsts;
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public final class StrictMath {

    public static final double E;

    public static final double PI;

    public static native double sin(double a);

    public static native double cos(double a);

    public static native double tan(double a);

    public static native double asin(double a);

    public static native double acos(double a);

    public static native double atan(double a);

    public static double toRadians(double angdeg);

    public static double toDegrees(double angrad);

    public static double exp(double a);

    public static native double log(double a);

    public static native double log10(double a);

    @IntrinsicCandidate
    public static native double sqrt(double a);

    public static double cbrt(double a);

    public static native double IEEEremainder(double f1, double f2);

    public static double ceil(double a);

    public static double floor(double a);

    public static double rint(double a);

    public static native double atan2(double y, double x);

    public static double pow(double a, double b);

    public static int round(float a);

    public static long round(double a);

    private static final class RandomNumberGeneratorHolder {
    }

    public static double random();

    public static int addExact(int x, int y);

    public static long addExact(long x, long y);

    public static int subtractExact(int x, int y);

    public static long subtractExact(long x, long y);

    public static int multiplyExact(int x, int y);

    public static long multiplyExact(long x, int y);

    public static long multiplyExact(long x, long y);

    @Pure
    public static int incrementExact(int a);

    @Pure
    public static long incrementExact(long a);

    @Pure
    public static int decrementExact(int a);

    @Pure
    public static long decrementExact(long a);

    @Pure
    public static int negateExact(int a);

    @Pure
    public static long negateExact(long a);

    public static int toIntExact(long value);

    public static long multiplyFull(int x, int y);

    public static long multiplyHigh(long x, long y);

    public static int floorDiv(int x, int y);

    public static long floorDiv(long x, int y);

    public static long floorDiv(long x, long y);

    public static int floorMod(int x, int y);

    public static int floorMod(long x, int y);

    public static long floorMod(long x, long y);

    public static int abs(int a);

    @Pure
    public static int absExact(int a);

    public static long abs(long a);

    @Pure
    public static long absExact(long a);

    public static float abs(float a);

    public static double abs(double a);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static int max(int a, int b);

    @Pure
    @StaticallyExecutable
    public static long max(long a, long b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static float max(float a, float b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static double max(double a, double b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static int min(int a, int b);

    @Pure
    @StaticallyExecutable
    public static long min(long a, long b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static float min(float a, float b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static double min(double a, double b);

    public static double fma(double a, double b, double c);

    public static float fma(float a, float b, float c);

    public static double ulp(double d);

    public static float ulp(float f);

    public static double signum(double d);

    public static float signum(float f);

    public static native double sinh(double x);

    public static native double cosh(double x);

    public static native double tanh(double x);

    public static double hypot(double x, double y);

    public static native double expm1(double x);

    public static native double log1p(double x);

    public static double copySign(double magnitude, double sign);

    public static float copySign(float magnitude, float sign);

    public static int getExponent(float f);

    public static int getExponent(double d);

    public static double nextAfter(double start, double direction);

    public static float nextAfter(float start, double direction);

    public static double nextUp(double d);

    public static float nextUp(float f);

    public static double nextDown(double d);

    public static float nextDown(float f);

    public static double scalb(double d, int scaleFactor);

    public static float scalb(float f, int scaleFactor);
}
