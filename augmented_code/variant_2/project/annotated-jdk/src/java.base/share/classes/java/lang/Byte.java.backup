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
import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
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
import static java.lang.constant.ConstantDescs.CD_byte;
import static java.lang.constant.ConstantDescs.CD_int;
import static java.lang.constant.ConstantDescs.DEFAULT_NAME;

@AnnotatedFor({ "index", "interning", "nullness", "signedness", "value" })
@jdk.internal.ValueBased
public final class Byte extends Number implements Comparable<Byte>, Constable {

    @IntVal(-128)
    public static final byte MIN_VALUE;

    @Positive
    @IntVal(127)
    public static final byte MAX_VALUE;

    @SuppressWarnings("unchecked")
    public static final Class<Byte> TYPE;

    @SideEffectFree
    @ArrayLen({ 1, 2, 3, 4 })
    public static String toString(byte b);

    @Override
    public Optional<DynamicConstantDesc<Byte>> describeConstable();

    private static class ByteCache {
    }

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @Interned
    @NewObject
    @PolyIndex
    @PolySigned
    @PolyValue
    public static Byte valueOf(@PolyIndex @PolySigned @PolyValue byte b);

    @Pure
    @StaticallyExecutable
    public static byte parseByte(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    public static byte parseByte(String s) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    @Interned
    @NewObject
    public static Byte valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    @Interned
    @NewObject
    public static Byte valueOf(String s) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    @NewObject
    public static Byte decode(String nm) throws NumberFormatException;

    @StaticallyExecutable
    @Deprecated()
    @PolyIndex
    @PolySigned
    @PolyValue
    public Byte(@PolyIndex @PolySigned @PolyValue byte value) {
    }

    @StaticallyExecutable
    @Deprecated()
    public Byte(String s) throws NumberFormatException {
    }

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyIndex
    @PolySigned
    @PolyValue
    public byte byteValue(@PolyIndex @PolySigned @PolyValue Byte this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolySigned
    @PolyValue
    public short shortValue(@PolyIndex @PolySigned @PolyValue Byte this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolySigned
    @PolyValue
    public int intValue(@PolyIndex @PolySigned @PolyValue Byte this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolySigned
    @PolyValue
    public long longValue(@PolyIndex @PolySigned @PolyValue Byte this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public float floatValue(@PolyValue Byte this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public double doubleValue(@PolyValue Byte this);

    @SideEffectFree
    @ArrayLen({ 1, 2, 3, 4 })
    public String toString();

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(byte value);

    @Pure
    @StaticallyExecutable
    public boolean equals(@Nullable Object obj);

    @Pure
    @StaticallyExecutable
    public int compareTo(Byte anotherByte);

    @Pure
    @StaticallyExecutable
    public static int compare(byte x, byte y);

    @Pure
    @StaticallyExecutable
    public static int compareUnsigned(@Unsigned byte x, @Unsigned byte y);

    @Pure
    @StaticallyExecutable
    @SignedPositive
    @NonNegative
    public static int toUnsignedInt(@UnknownSignedness byte x);

    @Pure
    @StaticallyExecutable
    @SignedPositive
    @NonNegative
    public static long toUnsignedLong(@UnknownSignedness byte x);

    @Positive
    @IntVal(8)
    public static final int SIZE;

    @Positive
    @IntVal(1)
    public static final int BYTES;
}
