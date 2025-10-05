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
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "nullness", "value" })
@jdk.internal.ValueBased
public final class Float extends Number implements Comparable<Float>, Constable, ConstantDesc {

    public static final float POSITIVE_INFINITY;

    public static final float NEGATIVE_INFINITY;

    public static final float NaN;

    public static final float MAX_VALUE;

    public static final float MIN_NORMAL;

    public static final float MIN_VALUE;

    @IntVal(127)
    public static final int MAX_EXPONENT;

    @IntVal(-126)
    public static final int MIN_EXPONENT;

    @IntVal(32)
    public static final int SIZE;

    @IntVal(4)
    public static final int BYTES;

    @SuppressWarnings("unchecked")
    public static final Class<Float> TYPE;

    @SideEffectFree
    @StaticallyExecutable
    public static String toString(float f);

    @SideEffectFree
    @StaticallyExecutable
    public static String toHexString(float f);

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Float valueOf(String s) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @IntrinsicCandidate
    @NewObject
    @PolyValue
    public static Float valueOf(@PolyValue float f);

    @Pure
    @StaticallyExecutable
    public static float parseFloat(String s) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    public static boolean isNaN(float v);

    @Pure
    @StaticallyExecutable
    public static boolean isInfinite(float v);

    @StaticallyExecutable
    public static boolean isFinite(float f);

    @StaticallyExecutable
    @Deprecated()
    @PolyValue
    public Float(@PolyValue float value) {
    }

    @StaticallyExecutable
    @Deprecated()
    @PolyValue
    public Float(@PolyValue double value) {
    }

    @StaticallyExecutable
    @Deprecated()
    public Float(String s) throws NumberFormatException {
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
    public byte byteValue(@PolyValue Float this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public short shortValue(@PolyValue Float this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public int intValue(@PolyValue Float this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public long longValue(@PolyValue Float this);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyValue
    public float floatValue(@PolyValue Float this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public double doubleValue(@PolyValue Float this);

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(float value);

    @Pure
    @StaticallyExecutable
    public boolean equals(@Nullable Object obj);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static int floatToIntBits(float value);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static native int floatToRawIntBits(float value);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static native float intBitsToFloat(int bits);

    @Pure
    @StaticallyExecutable
    public int compareTo(Float anotherFloat);

    @Pure
    @StaticallyExecutable
    public static int compare(float f1, float f2);

    @Pure
    @StaticallyExecutable
    public static float sum(float a, float b);

    @Pure
    @StaticallyExecutable
    public static float max(float a, float b);

    @Pure
    @StaticallyExecutable
    public static float min(float a, float b);

    @Override
    public Optional<Float> describeConstable();

    @Override
    public Float resolveConstantDesc(MethodHandles.Lookup lookup);
}
