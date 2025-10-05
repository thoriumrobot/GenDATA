/*
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.lock.qual.NewObject;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.DoubleVal;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.invoke.MethodHandles;
import java.lang.constant.Constable;
import java.lang.constant.ConstantDesc;
import java.util.Optional;
import jdk.internal.math.FloatingDecimal;
import jdk.internal.math.DoubleConsts;
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "nullness", "index", "value" })
@jdk.internal.ValueBased
public final class Double extends Number implements Comparable<Double>, Constable, ConstantDesc {

    public static final double POSITIVE_INFINITY;

    public static final double NEGATIVE_INFINITY;

    public static final double NaN;

    @DoubleVal(0x1.fffffffffffffP+1023)
    public static final double MAX_VALUE;

    public static final double MIN_NORMAL;

    @DoubleVal(0x0.0000000000001P-1022)
    public static final double MIN_VALUE;

    @IntVal(1023)
    public static final int MAX_EXPONENT;

    @IntVal(-1022)
    public static final int MIN_EXPONENT;

    @IntVal(64)
    public static final int SIZE;

    @IntVal(8)
    public static final int BYTES;

    @SuppressWarnings("unchecked")
    public static final Class<Double> TYPE;

    @SideEffectFree
    @StaticallyExecutable
    public static String toString(double d);

    @SideEffectFree
    @StaticallyExecutable
    public static String toHexString(double d);

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Double valueOf(String s) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @IntrinsicCandidate
    @NewObject
    @PolyValue
    public static Double valueOf(@PolyValue double d);

    @Pure
    @StaticallyExecutable
    public static double parseDouble(String s) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    public static boolean isNaN(double v);

    @Pure
    @StaticallyExecutable
    public static boolean isInfinite(double v);

    @Pure
    @StaticallyExecutable
    public static boolean isFinite(double d);

    @StaticallyExecutable
    @Deprecated()
    public Double(double value) {
    }

    @StaticallyExecutable
    @Deprecated()
    public Double(String s) throws NumberFormatException {
    }

    @Pure
    @StaticallyExecutable
    public boolean isNaN();

    @Pure
    @StaticallyExecutable
    public boolean isInfinite();

    @SideEffectFree
    @StaticallyExecutable
    public String toString();

    @Pure
    @StaticallyExecutable
    @PolyValue
    public byte byteValue(@PolyValue Double this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public short shortValue(@PolyValue Double this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public int intValue(@PolyValue Double this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public long longValue(@PolyValue Double this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public float floatValue(@PolyValue Double this);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyValue
    public double doubleValue(@PolyValue Double this);

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(double value);

    @Pure
    @StaticallyExecutable
    public boolean equals(@Nullable Object obj);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static long doubleToLongBits(double value);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static native long doubleToRawLongBits(double value);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static native double longBitsToDouble(long bits);

    @Pure
    @StaticallyExecutable
    public int compareTo(Double anotherDouble);

    @Pure
    @StaticallyExecutable
    public static int compare(double d1, double d2);

    @Pure
    @StaticallyExecutable
    public static double sum(double a, double b);

    @Pure
    @StaticallyExecutable
    public static double max(double a, double b);

    @Pure
    @StaticallyExecutable
    public static double min(double a, double b);

    @Override
    public Optional<Double> describeConstable();

    @Override
    public Double resolveConstantDesc(MethodHandles.Lookup lookup);
}
