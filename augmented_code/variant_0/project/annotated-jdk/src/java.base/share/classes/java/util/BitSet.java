/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.ByteOrder;
    @Positive
import java.nio.LongBuffer;
    @Positive
import java.util.function.IntConsumer;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.StreamSupport;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class BitSet implements Cloneable, java.io.Serializable {

    @Positive
    public BitSet() {
    @Positive
    }

    @Positive
    public BitSet(@NonNegative int nbits) {
    @Positive
    }

    @Positive
    public static BitSet valueOf(long[] longs);

    @Positive
    public static BitSet valueOf(LongBuffer lb);

    @Positive
    public static BitSet valueOf(byte[] bytes);

    @Positive
    public static BitSet valueOf(ByteBuffer bb);

    @Positive
    public byte[] toByteArray();

    @Positive
    public long[] toLongArray();

    @Positive
    public void flip(@GuardSatisfied BitSet this, @NonNegative int bitIndex);

    @Positive
    public void flip(@GuardSatisfied BitSet this, @NonNegative int fromIndex, @NonNegative int toIndex);

    @Positive
    public void set(@GuardSatisfied BitSet this, @NonNegative int bitIndex);

    @Positive
    public void set(@GuardSatisfied BitSet this, @NonNegative int bitIndex, boolean value);

    @Positive
    public void set(@GuardSatisfied BitSet this, @NonNegative int fromIndex, @NonNegative int toIndex);

    @Positive
    public void set(@GuardSatisfied BitSet this, @NonNegative int fromIndex, @NonNegative int toIndex, boolean value);

    @Positive
    public void clear(@GuardSatisfied BitSet this, @NonNegative int bitIndex);

    @Positive
    public void clear(@GuardSatisfied BitSet this, @NonNegative int fromIndex, @NonNegative int toIndex);

    @Positive
    public void clear(@GuardSatisfied BitSet this);

    @Positive
    @Pure
    @Positive
    public boolean get(@GuardSatisfied BitSet this, @NonNegative int bitIndex);

    @Positive
    @Pure
    @Positive
    public BitSet get(@GuardSatisfied BitSet this, @NonNegative int fromIndex, @NonNegative int toIndex);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int nextSetBit(@GuardSatisfied BitSet this, @NonNegative int fromIndex);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int nextClearBit(@GuardSatisfied BitSet this, @NonNegative int fromIndex);

    @Positive
    @GTENegativeOne
    @Positive
    public int previousSetBit(@GTENegativeOne int fromIndex);

    @Positive
    @GTENegativeOne
    @Positive
    public int previousClearBit(@GTENegativeOne int fromIndex);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int length(@GuardSatisfied BitSet this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied BitSet this);

    @Positive
    @Pure
    @Positive
    public boolean intersects(@GuardSatisfied BitSet this, @GuardSatisfied BitSet set);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int cardinality(@GuardSatisfied BitSet this);

    @Positive
    public void and(@GuardSatisfied BitSet this, BitSet set);

    @Positive
    public void or(@GuardSatisfied BitSet this, BitSet set);

    @Positive
    public void xor(@GuardSatisfied BitSet this, BitSet set);

    @Positive
    public void andNot(@GuardSatisfied BitSet this, BitSet set);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied BitSet this);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied BitSet this);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied BitSet this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied BitSet this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied BitSet this);

    @Positive
    public IntStream stream();
    @Positive
}
