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
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "nullness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Float extends Number implements Comparable<Float>, Constable, ConstantDesc {

    @Positive
    public static final float POSITIVE_INFINITY;

    @Positive
    public static final float NEGATIVE_INFINITY;

    @Positive
    public static final float NaN;

    @Positive
    public static final float MAX_VALUE;

    @Positive
    public static final float MIN_NORMAL;

    @Positive
    public static final float MIN_VALUE;

    @Positive
    @IntVal(127)
    @Positive
    public static final int MAX_EXPONENT;

    @Positive
    @IntVal(-126)
    @Positive
    public static final int MIN_EXPONENT;

    @Positive
    @IntVal(32)
    @Positive
    public static final int SIZE;

    @Positive
    @IntVal(4)
    @Positive
    public static final int BYTES;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Float> TYPE;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toString(float f);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toHexString(float f);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Float valueOf(String s) throws NumberFormatException;

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
    public static Float valueOf(@PolyValue float f);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static float parseFloat(String s) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isNaN(float v);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isInfinite(float v);

    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isFinite(float f);

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @PolyValue
    @Positive
    public Float(@PolyValue float value) {
    @Positive
    }

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @PolyValue
    @Positive
    public Float(@PolyValue double value) {
    @Positive
    }

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Float(String s) throws NumberFormatException {
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
    public byte byteValue(@PolyValue Float this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public short shortValue(@PolyValue Float this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public int intValue(@PolyValue Float this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public long longValue(@PolyValue Float this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue Float this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue Float this);

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
    public static int hashCode(float value);

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
    public static int floatToIntBits(float value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static native int floatToRawIntBits(float value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static native float intBitsToFloat(int bits);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(Float anotherFloat);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(float f1, float f2);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static float sum(float a, float b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static float max(float a, float b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static float min(float a, float b);

    @Positive
    @Override
    @Positive
    public Optional<Float> describeConstable();

    @Positive
    @Override
    @Positive
    public Float resolveConstantDesc(MethodHandles.Lookup lookup);
    @Positive
}

// CFWR semantic augmentation - variant 1
