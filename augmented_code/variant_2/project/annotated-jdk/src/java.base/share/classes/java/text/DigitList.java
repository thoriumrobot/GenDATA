/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.math.RoundingMode;
    @Positive
import jdk.internal.math.FloatingDecimal;

    @Positive
final class DigitList implements Cloneable {

    @Positive
    public static final int MAX_COUNT;

    @Positive
    public int decimalAt;

    @Positive
    public int count;

    @Positive
    public char[] digits;

    @Positive
    boolean isZero();

    @Positive
    void setRoundingMode(RoundingMode r);

    @Positive
    public void clear();

    @Positive
    public void append(char digit);

    @Positive
    public final double getDouble();

    @Positive
    public final long getLong();

    @Positive
    public final BigDecimal getBigDecimal();

    @Positive
    boolean fitsIntoLong(boolean isPositive, boolean ignoreNegativeZero);

    @Positive
    final void set(boolean isNegative, double source, int maximumFractionDigits);

    @Positive
    final void set(boolean isNegative, double source, int maximumDigits, boolean fixedPoint);

    @Positive
    final void set(boolean isNegative, long source);

    @Positive
    final void set(boolean isNegative, long source, int maximumDigits);

    @Positive
    final void set(boolean isNegative, BigDecimal source, int maximumDigits, boolean fixedPoint);

    @Positive
    final void set(boolean isNegative, BigInteger source, int maximumDigits);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public Object clone();

    @Positive
    public String toString();
    @Positive
}
