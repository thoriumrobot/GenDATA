/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import jdk.internal.math.FloatingDecimal;

final class DigitList implements Cloneable {

    public static final int MAX_COUNT;

    public int decimalAt;

    public int count;

    public char[] digits;

    boolean isZero();

    void setRoundingMode(RoundingMode r);

    public void clear();

    public void append(char digit);

    public final double getDouble();

    public final long getLong();

    public final BigDecimal getBigDecimal();

    boolean fitsIntoLong(boolean isPositive, boolean ignoreNegativeZero);

    final void set(boolean isNegative, double source, int maximumFractionDigits);

    final void set(boolean isNegative, double source, int maximumDigits, boolean fixedPoint);

    final void set(boolean isNegative, long source);

    final void set(boolean isNegative, long source, int maximumDigits);

    final void set(boolean isNegative, BigDecimal source, int maximumDigits, boolean fixedPoint);

    final void set(boolean isNegative, BigInteger source, int maximumDigits);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public Object clone();

    public String toString();
}
