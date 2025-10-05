/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.lock.qual.NewObject;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.DoubleVal;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.ConstantDesc;
    @Positive
import java.util.Optional;
    @Positive
import jdk.internal.math.FloatingDecimal;
    @Positive
import jdk.internal.math.DoubleConsts;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "nullness", "index", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Double extends Number implements Comparable<Double>, Constable, ConstantDesc {

    @Positive
    public static final double POSITIVE_INFINITY;

    @Positive
    public static final double NEGATIVE_INFINITY;

    @Positive
    public static final double NaN;

    @Positive
    @DoubleVal(0x1.fffffffffffffP+1023)
    @Positive
    public static final double MAX_VALUE;

    @Positive
    public static final double MIN_NORMAL;

    @Positive
    @DoubleVal(0x0.0000000000001P-1022)
    @Positive
    public static final double MIN_VALUE;

    @Positive
    @IntVal(1023)
    @Positive
    public static final int MAX_EXPONENT;

    @Positive
    @IntVal(-1022)
    @Positive
    public static final int MIN_EXPONENT;

    @Positive
    @IntVal(64)
    @Positive
    public static final int SIZE;

    @Positive
    @IntVal(8)
    @Positive
    public static final int BYTES;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Double> TYPE;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toString(double d);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toHexString(double d);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Double valueOf(String s) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NewObject
    @Positive
    @PolyValue
    @Positive
    public static Double valueOf(@PolyValue double d);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static double parseDouble(String s) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isNaN(double v);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isInfinite(double v);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isFinite(double d);

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Double(double value) {
    @Positive
    }

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Double(String s) throws NumberFormatException {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean isNaN();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean isInfinite();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public byte byteValue(@PolyValue Double this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public short shortValue(@PolyValue Double this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public int intValue(@PolyValue Double this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public long longValue(@PolyValue Double this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue Double this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue Double this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int hashCode(double value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static long doubleToLongBits(double value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static native long doubleToRawLongBits(double value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static native double longBitsToDouble(long bits);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(Double anotherDouble);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(double d1, double d2);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static double sum(double a, double b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static double max(double a, double b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static double min(double a, double b);

    @Positive
    @Override
    @Positive
    public Optional<Double> describeConstable();

    @Positive
    @Override
    @Positive
    public Double resolveConstantDesc(MethodHandles.Lookup lookup);
    @Positive
}
