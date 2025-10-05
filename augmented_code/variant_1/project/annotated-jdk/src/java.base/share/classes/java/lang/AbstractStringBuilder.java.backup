/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.math.FloatingDecimal;
import java.util.Arrays;
import java.util.Spliterator;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;
import jdk.internal.util.ArraysSupport;
import static java.lang.String.COMPACT_STRINGS;
import static java.lang.String.UTF16;
import static java.lang.String.LATIN1;
import static java.lang.String.checkIndex;
import static java.lang.String.checkOffset;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
abstract class AbstractStringBuilder implements Appendable, CharSequence {

    int compareTo(AbstractStringBuilder another);

    @Pure
    @Override
    @NonNegative
    public int length(@GuardSatisfied AbstractStringBuilder this);

    @NonNegative
    public int capacity();

    public void ensureCapacity(@NonNegative int minimumCapacity);

    public void trimToSize();

    public void setLength(@NonNegative int newLength);

    @Override
    public char charAt(@NonNegative int index);

    public int codePointAt(@NonNegative int index);

    public int codePointBefore(@Positive int index);

    @NonNegative
    public int codePointCount(@NonNegative int beginIndex, @NonNegative int endIndex);

    @NonNegative
    public int offsetByCodePoints(@NonNegative int index, int codePointOffset);

    public void getChars(@NonNegative int srcBegin, @NonNegative int srcEnd, char[] dst, @IndexOrHigh({ "#3" }) int dstBegin);

    public void setCharAt(@NonNegative int index, char ch);

    public AbstractStringBuilder append(@GuardSatisfied @Nullable Object obj);

    public AbstractStringBuilder append(@Nullable String str);

    public AbstractStringBuilder append(@Nullable StringBuffer sb);

    AbstractStringBuilder append(AbstractStringBuilder asb);

    @Override
    public AbstractStringBuilder append(@Nullable CharSequence s);

    @Override
    public AbstractStringBuilder append(@Nullable CharSequence s, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    public AbstractStringBuilder append(char[] str);

    public AbstractStringBuilder append(char[] str, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    public AbstractStringBuilder append(boolean b);

    @Override
    public AbstractStringBuilder append(char c);

    public AbstractStringBuilder append(int i);

    public AbstractStringBuilder append(long l);

    public AbstractStringBuilder append(float f);

    public AbstractStringBuilder append(double d);

    public AbstractStringBuilder delete(@NonNegative int start, @NonNegative int end);

    public AbstractStringBuilder appendCodePoint(int codePoint);

    public AbstractStringBuilder deleteCharAt(@NonNegative int index);

    public AbstractStringBuilder replace(@NonNegative int start, @NonNegative int end, String str);

    public String substring(@NonNegative int start);

    @Override
    public CharSequence subSequence(@NonNegative int start, @NonNegative int end);

    public String substring(@NonNegative int start, @NonNegative int end);

    public AbstractStringBuilder insert(@NonNegative int index, char[] str, @LTLengthOf(value = { "#2" }, offset = { "#4 - 1" }) @NonNegative int offset, @IndexOrHigh({ "#2" }) int len);

    public AbstractStringBuilder insert(@NonNegative int offset, @GuardSatisfied @Nullable Object obj);

    public AbstractStringBuilder insert(@NonNegative int offset, @Nullable String str);

    public AbstractStringBuilder insert(@NonNegative int offset, char[] str);

    public AbstractStringBuilder insert(@NonNegative int dstOffset, @Nullable CharSequence s);

    public AbstractStringBuilder insert(@NonNegative int dstOffset, @Nullable CharSequence s, @IndexOrHigh({ "#2" }) int start, @IndexOrHigh({ "#2" }) int end);

    public AbstractStringBuilder insert(@NonNegative int offset, boolean b);

    public AbstractStringBuilder insert(@NonNegative int offset, char c);

    public AbstractStringBuilder insert(@NonNegative int offset, int i);

    public AbstractStringBuilder insert(@NonNegative int offset, long l);

    public AbstractStringBuilder insert(@NonNegative int offset, float f);

    public AbstractStringBuilder insert(@NonNegative int offset, double d);

    @Pure
    @GTENegativeOne
    public int indexOf(@GuardSatisfied AbstractStringBuilder this, String str);

    @Pure
    @GTENegativeOne
    public int indexOf(@GuardSatisfied AbstractStringBuilder this, String str, int fromIndex);

    @Pure
    @GTENegativeOne
    public int lastIndexOf(@GuardSatisfied AbstractStringBuilder this, String str);

    @Pure
    @GTENegativeOne
    public int lastIndexOf(@GuardSatisfied AbstractStringBuilder this, String str, int fromIndex);

    public AbstractStringBuilder reverse();

    @SideEffectFree
    @Override
    public abstract String toString(@GuardSatisfied AbstractStringBuilder this);

    @Override
    public IntStream chars();

    @Override
    public IntStream codePoints();

    final byte[] getValue();

    void getBytes(byte[] dst, int dstBegin, byte coder);

    void initBytes(char[] value, int off, int len);

    final byte getCoder();

    final boolean isLatin1();
}
