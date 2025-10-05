/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyIndex;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.lock.qual.NewObject;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.checker.signedness.qual.Unsigned;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
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
import jdk.internal.misc.CDS;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.DynamicConstantDesc;
    @Positive
import java.util.Optional;
    @Positive
import static java.lang.constant.ConstantDescs.BSM_EXPLICIT_CAST;
    @Positive
import static java.lang.constant.ConstantDescs.CD_int;
    @Positive
import static java.lang.constant.ConstantDescs.CD_short;
    @Positive
import static java.lang.constant.ConstantDescs.DEFAULT_NAME;

    @Positive
@AnnotatedFor({ "nullness", "index", "signedness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Short extends Number implements Comparable<Short>, Constable {

    @Positive
    @IntVal(-32768)
    @Positive
    public static final short MIN_VALUE;

    @Positive
    @Positive
    @Positive
    @IntVal(32767)
    @Positive
    public static final short MAX_VALUE;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Short> TYPE;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLen({ 1, 2, 3, 4, 5, 6 })
    @Positive
    public static String toString(short s);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static short parseShort(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static short parseShort(String s) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Short valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Short valueOf(String s) throws NumberFormatException;

    @Positive
    @Override
    @Positive
    public Optional<DynamicConstantDesc<Short>> describeConstable();

    @Positive
    private static class ShortCache {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NewObject
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public static Short valueOf(@PolyIndex @PolySigned @PolyValue short s);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Short decode(String nm) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public Short(@PolyIndex @PolySigned @PolyValue short value) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Short(String s) throws NumberFormatException {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolyValue
    @Positive
    public byte byteValue(@PolyIndex @PolyValue Short this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public short shortValue(@PolyIndex @PolySigned @PolyValue Short this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public int intValue(@PolyIndex @PolySigned @PolyValue Short this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public long longValue(@PolyIndex @PolySigned @PolyValue Short this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue Short this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue Short this);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLen({ 1, 2, 3, 4, 5, 6 })
    @Positive
    public String toString();

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
    public static int hashCode(short value);

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
    public int compareTo(Short anotherShort);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(short x, short y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compareUnsigned(@Unsigned short x, @Unsigned short y);

    @Positive
    @Positive
    @Positive
    @IntVal(16)
    @Positive
    public static final int SIZE;

    @Positive
    @IntVal(2)
    @Positive
    public static final int BYTES;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static short reverseBytes(short i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @NonNegative
    @Positive
    @SignedPositive
    @Positive
    public static int toUnsignedInt(@UnknownSignedness short x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @NonNegative
    @Positive
    @SignedPositive
    @Positive
    public static long toUnsignedLong(@UnknownSignedness short x);
    @Positive
}
