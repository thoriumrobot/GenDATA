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

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyIndex;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.lock.qual.NewObject;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.checker.signedness.qual.SignednessGlb;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.checker.signedness.qual.Unsigned;
import org.checkerframework.common.value.qual.ArrayLenRange;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.annotation.Native;
import java.lang.invoke.MethodHandles;
import java.lang.constant.Constable;
import java.lang.constant.ConstantDesc;
import java.math.*;
import java.util.Objects;
import java.util.Optional;
import jdk.internal.misc.CDS;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import static java.lang.String.COMPACT_STRINGS;
import static java.lang.String.LATIN1;
import static java.lang.String.UTF16;

@AnnotatedFor({ "nullness", "index", "signedness", "value" })
@jdk.internal.ValueBased
public final class Long extends Number implements Comparable<Long>, Constable, ConstantDesc {

    @Native
    @SignednessGlb
    @IntVal(0x8000000000000000L)
    public static final long MIN_VALUE;

    @Native
    @SignedPositive
    @IntVal(0x7fffffffffffffffL)
    public static final long MAX_VALUE;

    @SuppressWarnings("unchecked")
    public static final Class<Long> TYPE;

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1)
    public static String toString(long i, @Positive @IntRange(from = 2, to = 36) int radix);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1)
    public static String toUnsignedString(@Unsigned long i, @Positive @IntRange(from = 2, to = 36) int radix);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1, to = 16)
    public static String toHexString(@UnknownSignedness long i);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1, to = 22)
    public static String toOctalString(@Unsigned long i);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1, to = 64)
    public static String toBinaryString(@Unsigned long i);

    static String toUnsignedString0(@Unsigned long val, @IntVal({ 1, 2, 3, 4, 5 }) int shift);

    static String fastUUID(long lsb, long msb);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1, to = 20)
    public static String toString(long i);

    @SideEffectFree
    @StaticallyExecutable
    public static String toUnsignedString(@Unsigned long i);

    static int getChars(long i, int index, byte[] buf);

    static int stringSize(long x);

    @Pure
    @StaticallyExecutable
    public static long parseLong(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    public static long parseLong(CharSequence s, int beginIndex, int endIndex, @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    public static long parseLong(String s) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    @Unsigned
    public static long parseUnsignedLong(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    @Unsigned
    public static long parseUnsignedLong(CharSequence s, int beginIndex, int endIndex, @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Pure
    @StaticallyExecutable
    @Unsigned
    public static long parseUnsignedLong(String s) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Long valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @NewObject
    public static Long valueOf(String s) throws NumberFormatException;

    private static class LongCache {
    }

    @SideEffectFree
    @StaticallyExecutable
    @IntrinsicCandidate
    @NewObject
    @PolySigned
    @PolyValue
    public static Long valueOf(@PolySigned @PolyValue long l);

    @SideEffectFree
    @StaticallyExecutable
    public static Long decode(String nm) throws NumberFormatException;

    @SideEffectFree
    @StaticallyExecutable
    @Deprecated()
    @PolyIndex
    @PolySigned
    @PolyValue
    public Long(@PolyIndex @PolySigned @PolyValue long value) {
    }

    @SideEffectFree
    @StaticallyExecutable
    @Deprecated()
    public Long(String s) throws NumberFormatException {
    }

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolyValue
    public byte byteValue(@PolyIndex @PolyValue Long this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolyValue
    public short shortValue(@PolyIndex @PolyValue Long this);

    @Pure
    @StaticallyExecutable
    @PolyIndex
    @PolyValue
    public int intValue(@PolyIndex @PolyValue Long this);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyIndex
    @PolySigned
    @PolyValue
    public long longValue(@PolyIndex @PolySigned @PolyValue Long this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public float floatValue(@PolyValue Long this);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public double doubleValue(@PolyValue Long this);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLenRange(from = 1, to = 20)
    public String toString();

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(long value);

    @Pure
    @StaticallyExecutable
    public boolean equals(@Nullable Object obj);

    @SideEffectFree
    @StaticallyExecutable
    @Nullable
    public static Long getLong(@Nullable String nm);

    @SideEffectFree
    @StaticallyExecutable
    public static Long getLong(@Nullable String nm, long val);

    @SideEffectFree
    @StaticallyExecutable
    @PolyNull
    public static Long getLong(@Nullable String nm, @PolyNull Long val);

    @Pure
    @StaticallyExecutable
    public int compareTo(Long anotherLong);

    @Pure
    @StaticallyExecutable
    public static int compare(long x, long y);

    @Pure
    @StaticallyExecutable
    public static int compareUnsigned(@Unsigned long x, @Unsigned long y);

    @Pure
    @StaticallyExecutable
    @Unsigned
    public static long divideUnsigned(@Unsigned long dividend, @Unsigned long divisor);

    @Pure
    @StaticallyExecutable
    @Unsigned
    public static long remainderUnsigned(@Unsigned long dividend, @Unsigned long divisor);

    @Native
    @SignedPositive
    @IntVal(64)
    public static final int SIZE;

    @SignedPositive
    @IntVal(8)
    public static final int BYTES;

    @Pure
    @StaticallyExecutable
    public static long highestOneBit(@UnknownSignedness long i);

    @Pure
    @StaticallyExecutable
    public static long lowestOneBit(@UnknownSignedness long i);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @NonNegative
    @IntRange(from = 0, to = 64)
    public static int numberOfLeadingZeros(@UnknownSignedness long i);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @NonNegative
    @IntRange(from = 0, to = 64)
    public static int numberOfTrailingZeros(@UnknownSignedness long i);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @NonNegative
    public static int bitCount(@UnknownSignedness long i);

    @Pure
    @StaticallyExecutable
    @PolySigned
    public static long rotateLeft(@PolySigned long i, int distance);

    @Pure
    @StaticallyExecutable
    @PolySigned
    public static long rotateRight(@PolySigned long i, int distance);

    @Pure
    @StaticallyExecutable
    @SignednessGlb
    public static long reverse(@PolySigned long i);

    @Pure
    @StaticallyExecutable
    @GTENegativeOne
    public static int signum(long i);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @SignednessGlb
    public static long reverseBytes(@PolySigned long i);

    @Pure
    @StaticallyExecutable
    public static long sum(long a, long b);

    @Pure
    @StaticallyExecutable
    public static long max(long a, long b);

    @Pure
    @StaticallyExecutable
    public static long min(long a, long b);

    @Override
    public Optional<Long> describeConstable();

    @Override
    public Long resolveConstantDesc(MethodHandles.Lookup lookup);
}
