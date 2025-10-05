/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyIndex;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.lock.qual.NewObject;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.checker.signedness.qual.Unsigned;
import org.checkerframework.common.value.qual.ArrayLen;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.misc.CDS;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import java.lang.constant.Constable;
import java.lang.constant.DynamicConstantDesc;
import java.util.Optional;
import static java.lang.constant.ConstantDescs.BSM_EXPLICIT_CAST;
import static java.lang.constant.ConstantDescs.CD_int;
import static java.lang.constant.ConstantDescs.CD_short;
import static java.lang.constant.ConstantDescs.DEFAULT_NAME;

@AnnotatedFor({ "nullness", "index", "signedness", "value" })
@jdk.internal.ValueBased
public final class Short extends Number implements Comparable<Short>, Constable {

    @IntVal(-32768)
    public static final short MIN_VALUE;

    @Positive
    @IntVal(32767)
    public static final short MAX_VALUE;

    @SuppressWarnings("unchecked")
    public static final Class<Short> TYPE;

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLen({ 1, 2, 3, 4, 5, 6 })
    public static String toString(short s);

    @Pure
    @StaticallyExecutable
    public static short parseShort(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    public static short parseShort(String s) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Short valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Short valueOf(String s) throws NumberFormatException;

    @Override
    public Optional<DynamicConstantDesc<Short>> describeConstable();

    private static class ShortCache {
    }

    @SideEffectFree
    @StaticallyExecutable
    @IntrinsicCandidate
    @NewObject
    @PolyIndex
    @PolySigned
    @PolyValue
    public static Short valueOf(@PolyIndex @PolySigned @PolyValue short s);

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Short decode(String nm) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @Deprecated()
    @PolyIndex
    @PolySigned
    @PolyValue
    public Short(@PolyIndex @PolySigned @PolyValue short value) {
    }

    @SideEffectFree
    @StaticallyExecutable
    @Deprecated()
    public Short(String s) throws NumberFormatException {
    }

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolyValue
    public byte byteValue(@PolyIndex @PolyValue Short this);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyIndex
    @PolySigned
    @PolyValue
    public short shortValue(@PolyIndex @PolySigned @PolyValue Short this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolySigned
    @PolyValue
    public int intValue(@PolyIndex @PolySigned @PolyValue Short this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolySigned
    @PolyValue
    public long longValue(@PolyIndex @PolySigned @PolyValue Short this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public float floatValue(@PolyValue Short this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public double doubleValue(@PolyValue Short this);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLen({ 1, 2, 3, 4, 5, 6 })
    public String toString();

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(short value);

    @Pure
    @StaticallyExecutable
    public boolean equals(@Nullable Object obj);

    @Pure
    @StaticallyExecutable
    public int compareTo(Short anotherShort);

    @Pure
    @StaticallyExecutable
    public static int compare(short x, short y);

    @Pure
    @StaticallyExecutable
    public static int compareUnsigned(@Unsigned short x, @Unsigned short y);

    @Positive
    @IntVal(16)
    public static final int SIZE;

    @IntVal(2)
    public static final int BYTES;

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static short reverseBytes(short i);

    @Pure
    @StaticallyExecutable
    @NonNegative
    @SignedPositive
    public static int toUnsignedInt(@UnknownSignedness short x);

    @Pure
    @StaticallyExecutable
    @NonNegative
    @SignedPositive
    public static long toUnsignedLong(@UnknownSignedness short x);
}
