/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.math.FloatingDecimal;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.util.ArraysSupport;
    @Positive
import static java.lang.String.COMPACT_STRINGS;
    @Positive
import static java.lang.String.UTF16;
    @Positive
import static java.lang.String.LATIN1;
    @Positive
import static java.lang.String.checkIndex;
    @Positive
import static java.lang.String.checkOffset;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
abstract class AbstractStringBuilder implements Appendable, CharSequence {

    @Positive
    int compareTo(AbstractStringBuilder another);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @NonNegative
    @Positive
    public int length(@GuardSatisfied AbstractStringBuilder this);

    @Positive
    @NonNegative
    @Positive
    public int capacity();

    @Positive
    public void ensureCapacity(@NonNegative int minimumCapacity);

    @Positive
    public void trimToSize();

    @Positive
    public void setLength(@NonNegative int newLength);

    @Positive
    @Override
    @Positive
    public char charAt(@NonNegative int index);

    @Positive
    public int codePointAt(@NonNegative int index);

    @Positive
    public int codePointBefore(@Positive int index);

    @Positive
    @NonNegative
    @Positive
    public int codePointCount(@NonNegative int beginIndex, @NonNegative int endIndex);

    @Positive
    @NonNegative
    @Positive
    public int offsetByCodePoints(@NonNegative int index, int codePointOffset);

    @Positive
    public void getChars(@NonNegative int srcBegin, @NonNegative int srcEnd, char[] dst, @IndexOrHigh({ "#3" }) int dstBegin);

    @Positive
    public void setCharAt(@NonNegative int index, char ch);

    @Positive
    public AbstractStringBuilder append(@GuardSatisfied @Nullable Object obj);

    @Positive
    public AbstractStringBuilder append(@Nullable String str);

    @Positive
    public AbstractStringBuilder append(@Nullable StringBuffer sb);

    @Positive
    AbstractStringBuilder append(AbstractStringBuilder asb);

    @Positive
    @Override
    @Positive
    public AbstractStringBuilder append(@Nullable CharSequence s);

    @Positive
    @Override
    @Positive
    public AbstractStringBuilder append(@Nullable CharSequence s, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @Positive
    public AbstractStringBuilder append(char[] str);

    @Positive
    public AbstractStringBuilder append(char[] str, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Positive
    public AbstractStringBuilder append(boolean b);

    @Positive
    @Override
    @Positive
    public AbstractStringBuilder append(char c);

    @Positive
    public AbstractStringBuilder append(int i);

    @Positive
    public AbstractStringBuilder append(long l);

    @Positive
    public AbstractStringBuilder append(float f);

    @Positive
    public AbstractStringBuilder append(double d);

    @Positive
    public AbstractStringBuilder delete(@NonNegative int start, @NonNegative int end);

    @Positive
    public AbstractStringBuilder appendCodePoint(int codePoint);

    @Positive
    public AbstractStringBuilder deleteCharAt(@NonNegative int index);

    @Positive
    public AbstractStringBuilder replace(@NonNegative int start, @NonNegative int end, String str);

    @Positive
    public String substring(@NonNegative int start);

    @Positive
    @Override
    @Positive
    public CharSequence subSequence(@NonNegative int start, @NonNegative int end);

    @Positive
    public String substring(@NonNegative int start, @NonNegative int end);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int index, char[] str, @LTLengthOf(value = { "#2" }, offset = { "#4 - 1" }) @NonNegative int offset, @IndexOrHigh({ "#2" }) int len);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, @GuardSatisfied @Nullable Object obj);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, @Nullable String str);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, char[] str);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int dstOffset, @Nullable CharSequence s);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int dstOffset, @Nullable CharSequence s, @IndexOrHigh({ "#2" }) int start, @IndexOrHigh({ "#2" }) int end);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, boolean b);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, char c);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, int i);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, long l);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, float f);

    @Positive
    public AbstractStringBuilder insert(@NonNegative int offset, double d);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied AbstractStringBuilder this, String str);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied AbstractStringBuilder this, String str, int fromIndex);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied AbstractStringBuilder this, String str);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied AbstractStringBuilder this, String str, int fromIndex);

    @Positive
    public AbstractStringBuilder reverse();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public abstract String toString(@GuardSatisfied AbstractStringBuilder this);

    @Positive
    @Override
    @Positive
    public IntStream chars();

    @Positive
    @Override
    @Positive
    public IntStream codePoints();

    @Positive
    final byte[] getValue();

    @Positive
    void getBytes(byte[] dst, int dstBegin, byte coder);

    @Positive
    void initBytes(char[] value, int off, int len);

    @Positive
    final byte getCoder();

    @Positive
    final boolean isLatin1();
    @Positive
}

// CFWR semantic augmentation - variant 1
