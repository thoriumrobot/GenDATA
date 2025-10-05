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
package java.math;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import static java.math.BigDecimal.INFLATED;
import static java.math.BigInteger.LONG_MASK;
import java.util.Arrays;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
class MutableBigInteger {

    BigInteger toBigInteger(int sign);

    BigInteger toBigInteger();

    BigDecimal toBigDecimal(int sign, int scale);

    long toCompactValue(int sign);

    void clear();

    void reset();

    final int compare(MutableBigInteger b);

    final int compareHalf(MutableBigInteger b);

    final void normalize();

    int[] toIntArray();

    void setInt(int index, int val);

    void setValue(int[] val, int length);

    void copyValue(MutableBigInteger src);

    void copyValue(int[] val);

    boolean isOne();

    boolean isZero();

    boolean isEven();

    boolean isOdd();

    boolean isNormal();

    public String toString();

    void safeRightShift(int n);

    void rightShift(int n);

    void safeLeftShift(int n);

    void leftShift(int n);

    void add(MutableBigInteger addend);

    void addShifted(MutableBigInteger addend, int n);

    void addDisjoint(MutableBigInteger addend, int n);

    void addLower(MutableBigInteger addend, int n);

    int subtract(MutableBigInteger b);

    void multiply(MutableBigInteger y, MutableBigInteger z);

    void mul(int y, MutableBigInteger z);

    int divideOneWord(int divisor, MutableBigInteger quotient);

    MutableBigInteger divide(MutableBigInteger b, MutableBigInteger quotient);

    MutableBigInteger divide(MutableBigInteger b, MutableBigInteger quotient, boolean needRemainder);

    MutableBigInteger divideKnuth(MutableBigInteger b, MutableBigInteger quotient);

    MutableBigInteger divideKnuth(MutableBigInteger b, MutableBigInteger quotient, boolean needRemainder);

    MutableBigInteger divideAndRemainderBurnikelZiegler(MutableBigInteger b, MutableBigInteger quotient);

    long bitLength();

    long divide(long v, MutableBigInteger quotient);

    static long divWord(long n, int d);

    MutableBigInteger sqrt();

    MutableBigInteger hybridGCD(MutableBigInteger b);

    static int binaryGcd(int a, int b);

    MutableBigInteger mutableModInverse(MutableBigInteger p);

    MutableBigInteger modInverseMP2(int k);

    static int inverseMod32(int val);

    static long inverseMod64(long val);

    static MutableBigInteger modInverseBP2(MutableBigInteger mod, int k);

    static MutableBigInteger fixup(MutableBigInteger c, MutableBigInteger p, int k);

    MutableBigInteger euclidModInverse(int k);
}
