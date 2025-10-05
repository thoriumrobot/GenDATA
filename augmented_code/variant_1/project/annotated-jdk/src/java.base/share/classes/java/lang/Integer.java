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
import org.checkerframework.checker.index.qual.GTENegativeOne;
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
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.checker.signedness.qual.SignednessGlb;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.checker.signedness.qual.Unsigned;
    @Positive
import org.checkerframework.common.value.qual.ArrayLenRange;
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
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.lang.annotation.Native;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.ConstantDesc;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import jdk.internal.misc.CDS;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import static java.lang.String.COMPACT_STRINGS;
    @Positive
import static java.lang.String.LATIN1;
    @Positive
import static java.lang.String.UTF16;

    @Positive
@AnnotatedFor({ "index", "nullness", "lock", "signedness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Integer extends Number implements Comparable<Integer>, Constable, ConstantDesc {

    @Positive
    @Native
    @Positive
    @SignednessGlb
    @Positive
    @IntVal(0x80000000)
    @Positive
    public static final int MIN_VALUE;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    @IntVal(0x7fffffff)
    @Positive
    public static final int MAX_VALUE;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Integer> TYPE;

    @Positive
    @CFComment("@IntRange(2, 36) int radix: the method uses 10 if radix is outside the valid range, but that is still probably an error, and other methods (like many methods in Integer, and Byte.toString) do throw an exception if the radix is outside the valid range")
    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1)
    @Positive
    public static String toString(int i, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @CFComment("@IntRange(2, 36) int radix: see CFComment on toString")
    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toUnsignedString(@Unsigned int i, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 8)
    @Positive
    public static String toHexString(@UnknownSignedness int i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 11)
    @Positive
    public static String toOctalString(@Unsigned int i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 32)
    @Positive
    public static String toBinaryString(@Unsigned int i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @ArrayLenRange(from = 1, to = 11)
    @Positive
    public static String toString(int i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String toUnsignedString(@Unsigned int i);

    @Positive
    static int getChars(int i, int index, byte[] buf);

    @Positive
    static int stringSize(int x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int parseInt(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int parseInt(CharSequence s, int beginIndex, int endIndex, @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int parseInt(String s) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static int parseUnsignedInt(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static int parseUnsignedInt(CharSequence s, int beginIndex, int endIndex, @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static int parseUnsignedInt(String s) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Integer valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Integer valueOf(String s) throws NumberFormatException;

    @Positive
    private static class IntegerCache {
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
    public static Integer valueOf(@PolyIndex @PolySigned @PolyValue int i);

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
    public Integer(@PolyIndex @PolySigned @PolyValue int value) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Integer(String s) throws NumberFormatException {
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
    public byte byteValue(@PolyIndex @PolyValue Integer this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyIndex
    @Positive
    @PolyValue
    @Positive
    public short shortValue(@PolyIndex @PolyValue Integer this);

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
    public int intValue(@PolyIndex @PolySigned @PolyValue Integer this);

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
    public long longValue(@PolyIndex @PolySigned @PolyValue Integer this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue Integer this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue Integer this);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLenRange(from = 1, to = 11)
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
    public static int hashCode(int value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Nullable
    @Positive
    public static Integer getInteger(@Nullable String nm);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static Integer getInteger(@Nullable String nm, int val);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyNull
    @Positive
    public static Integer getInteger(@Nullable String nm, @PolyNull Integer val);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static Integer decode(String nm) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(Integer anotherInteger);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(int x, int y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compareUnsigned(@Unsigned int x, @Unsigned int y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @SignedPositive
    @Positive
    public static long toUnsignedLong(@UnknownSignedness int x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static int divideUnsigned(@Unsigned int dividend, @Unsigned int divisor);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Unsigned
    @Positive
    public static int remainderUnsigned(@Unsigned int dividend, @Unsigned int divisor);

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    @IntVal(32)
    @Positive
    public static final int SIZE;

    @Positive
    @SignedPositive
    @Positive
    @IntVal(4)
    @Positive
    public static final int BYTES;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int highestOneBit(@UnknownSignedness int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int lowestOneBit(@UnknownSignedness int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    @IntRange(from = 0, to = 32)
    @Positive
    public static int numberOfLeadingZeros(@UnknownSignedness int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    @IntRange(from = 0, to = 32)
    @Positive
    public static int numberOfTrailingZeros(@UnknownSignedness int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    public static int bitCount(@UnknownSignedness int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public static int rotateLeft(@PolySigned int i, int distance);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public static int rotateRight(@PolySigned int i, int distance);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @SignednessGlb
    @Positive
    public static int reverse(@PolySigned int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @GTENegativeOne
    @Positive
    public static int signum(int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @SignednessGlb
    @Positive
    public static int reverseBytes(@PolySigned int i);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int sum(int a, int b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int max(int a, int b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int min(int a, int b);

    @Positive
    @Override
    @Positive
    public Optional<Integer> describeConstable();

    @Positive
    @Override
    @Positive
    public Integer resolveConstantDesc(MethodHandles.Lookup lookup);
    @Positive
}
