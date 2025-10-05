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
package java.math;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import static java.math.BigDecimal.INFLATED;
    @Positive
import static java.math.BigInteger.LONG_MASK;
    @Positive
import java.util.Arrays;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class MutableBigInteger {

    @Positive
    BigInteger toBigInteger(int sign);

    @Positive
    BigInteger toBigInteger();

    @Positive
    BigDecimal toBigDecimal(int sign, int scale);

    @Positive
    long toCompactValue(int sign);

    @Positive
    void clear();

    @Positive
    void reset();

    @Positive
    final int compare(MutableBigInteger b);

    @Positive
    final int compareHalf(MutableBigInteger b);

    @Positive
    final void normalize();

    @Positive
    int[] toIntArray();

    @Positive
    void setInt(int index, int val);

    @Positive
    void setValue(int[] val, int length);

    @Positive
    void copyValue(MutableBigInteger src);

    @Positive
    void copyValue(int[] val);

    @Positive
    boolean isOne();

    @Positive
    boolean isZero();

    @Positive
    boolean isEven();

    @Positive
    boolean isOdd();

    @Positive
    boolean isNormal();

    @Positive
    public String toString();

    @Positive
    void safeRightShift(int n);

    @Positive
    void rightShift(int n);

    @Positive
    void safeLeftShift(int n);

    @Positive
    void leftShift(int n);

    @Positive
    void add(MutableBigInteger addend);

    @Positive
    void addShifted(MutableBigInteger addend, int n);

    @Positive
    void addDisjoint(MutableBigInteger addend, int n);

    @Positive
    void addLower(MutableBigInteger addend, int n);

    @Positive
    int subtract(MutableBigInteger b);

    @Positive
    void multiply(MutableBigInteger y, MutableBigInteger z);

    @Positive
    void mul(int y, MutableBigInteger z);

    @Positive
    int divideOneWord(int divisor, MutableBigInteger quotient);

    @Positive
    MutableBigInteger divide(MutableBigInteger b, MutableBigInteger quotient);

    @Positive
    MutableBigInteger divide(MutableBigInteger b, MutableBigInteger quotient, boolean needRemainder);

    @Positive
    MutableBigInteger divideKnuth(MutableBigInteger b, MutableBigInteger quotient);

    @Positive
    MutableBigInteger divideKnuth(MutableBigInteger b, MutableBigInteger quotient, boolean needRemainder);

    @Positive
    MutableBigInteger divideAndRemainderBurnikelZiegler(MutableBigInteger b, MutableBigInteger quotient);

    @Positive
    long bitLength();

    @Positive
    long divide(long v, MutableBigInteger quotient);

    @Positive
    static long divWord(long n, int d);

    @Positive
    MutableBigInteger sqrt();

    @Positive
    MutableBigInteger hybridGCD(MutableBigInteger b);

    @Positive
    static int binaryGcd(int a, int b);

    @Positive
    MutableBigInteger mutableModInverse(MutableBigInteger p);

    @Positive
    MutableBigInteger modInverseMP2(int k);

    @Positive
    static int inverseMod32(int val);

    @Positive
    static long inverseMod64(long val);

    @Positive
    static MutableBigInteger modInverseBP2(MutableBigInteger mod, int k);

    @Positive
    static MutableBigInteger fixup(MutableBigInteger c, MutableBigInteger p, int k);

    @Positive
    MutableBigInteger euclidModInverse(int k);
    @Positive
}

// CFWR semantic augmentation - variant 1
