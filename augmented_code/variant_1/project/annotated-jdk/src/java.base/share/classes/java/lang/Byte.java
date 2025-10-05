/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
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
import static java.lang.constant.ConstantDescs.CD_byte;
    @Positive
import static java.lang.constant.ConstantDescs.CD_int;
    @Positive
import static java.lang.constant.ConstantDescs.DEFAULT_NAME;

    @Positive
@AnnotatedFor({ "index", "interning", "nullness", "signedness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Byte extends Number implements Comparable<Byte>, Constable {

    @Positive
    @IntVal(-128)
    @Positive
    public static final byte MIN_VALUE;

    @Positive
    @Positive
    @Positive
    @IntVal(127)
    @Positive
    public static final byte MAX_VALUE;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Byte> TYPE;

    @Positive
    @SideEffectFree
    @Positive
    @ArrayLen({ 1, 2, 3, 4 })
    @Positive
    public static String toString(byte b);

    @Positive
    @Override
    @Positive
    public Optional<DynamicConstantDesc<Byte>> describeConstable();

    @Positive
    private static class ByteCache {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @Interned
    @Positive
    @NewObject
    @Positive
    @PolyIndex
    @Positive
    @PolySigned
    @Positive
    @PolyValue
    @Positive
    public static Byte valueOf(@PolyIndex @PolySigned @PolyValue byte b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static byte parseByte(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static byte parseByte(String s) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Interned
    @Positive
    @NewObject
    @Positive
    public static Byte valueOf(String s, @Positive @IntRange(from = 2, to = 36) int radix) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Interned
    @Positive
    @NewObject
    @Positive
    public static Byte valueOf(String s) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static Byte decode(String nm) throws NumberFormatException;

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
    public Byte(@PolyIndex @PolySigned @PolyValue byte value) {
    @Positive
    }

    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public Byte(String s) throws NumberFormatException {
    @Positive
    }

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
    public byte byteValue(@PolyIndex @PolySigned @PolyValue Byte this);

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
    public short shortValue(@PolyIndex @PolySigned @PolyValue Byte this);

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
    public int intValue(@PolyIndex @PolySigned @PolyValue Byte this);

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
    public long longValue(@PolyIndex @PolySigned @PolyValue Byte this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue Byte this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue Byte this);

    @Positive
    @SideEffectFree
    @Positive
    @ArrayLen({ 1, 2, 3, 4 })
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
    public static int hashCode(byte value);

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
    public int compareTo(Byte anotherByte);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(byte x, byte y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compareUnsigned(@Unsigned byte x, @Unsigned byte y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @SignedPositive
    @Positive
    @NonNegative
    @Positive
    public static int toUnsignedInt(@UnknownSignedness byte x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @SignedPositive
    @Positive
    @NonNegative
    @Positive
    public static long toUnsignedLong(@UnknownSignedness byte x);

    @Positive
    @Positive
    @Positive
    @IntVal(8)
    @Positive
    public static final int SIZE;

    @Positive
    @Positive
    @Positive
    @IntVal(1)
    @Positive
    public static final int BYTES;
    @Positive
}
