/*
    @Positive
 * Copyright (c) 1994, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
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
import org.checkerframework.common.aliasing.qual.Unique;
    @Positive
import org.checkerframework.common.aliasing.qual.LeakedToResult;
    @Positive
import org.checkerframework.common.aliasing.qual.NonLeaked;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Arrays;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "aliasing", "lock", "nullness", "index" })
    @Positive
public final class StringBuffer extends AbstractStringBuilder implements java.io.Serializable, Comparable<StringBuffer>, CharSequence {

    @Positive
    @IntrinsicCandidate
    @Positive
    @Unique
    @Positive
    public StringBuffer() {
    @Positive
    }

    @Positive
    @IntrinsicCandidate
    @Positive
    @Unique
    @Positive
    public StringBuffer(@NonNegative int capacity) {
    @Positive
    }

    @Positive
    @IntrinsicCandidate
    @Positive
    @Unique
    @Positive
    public StringBuffer(String str) {
    @Positive
    }

    @Positive
    @Unique
    @Positive
    public StringBuffer(CharSequence seq) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public synchronized int compareTo(StringBuffer another);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @NonNegative
    @Positive
    public synchronized int length(@GuardSatisfied StringBuffer this);

    @Positive
    @Override
    @Positive
    @NonNegative
    @Positive
    public synchronized int capacity();

    @Positive
    @Override
    @Positive
    public synchronized void ensureCapacity(int minimumCapacity);

    @Positive
    @Override
    @Positive
    public synchronized void trimToSize();

    @Positive
    @Override
    @Positive
    public synchronized void setLength(@NonNegative int newLength);

    @Positive
    @Override
    @Positive
    public synchronized char charAt(int index);

    @Positive
    @Override
    @Positive
    public synchronized int codePointAt(int index);

    @Positive
    @Override
    @Positive
    public synchronized int codePointBefore(int index);

    @Positive
    @Override
    @Positive
    public synchronized int codePointCount(int beginIndex, int endIndex);

    @Positive
    @Override
    @Positive
    public synchronized int offsetByCodePoints(int index, int codePointOffset);

    @Positive
    @Override
    @Positive
    public synchronized void getChars(int srcBegin, int srcEnd, char[] dst, @IndexOrHigh({ "#3" }) int dstBegin);

    @Positive
    @Override
    @Positive
    public synchronized void setCharAt(int index, char ch);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable Object obj);

    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable String str);

    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable StringBuffer sb);

    @Positive
    @Override
    @Positive
    synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked AbstractStringBuilder asb);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable CharSequence s);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable CharSequence s, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked char[] str);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked char[] str, @IndexOrHigh({ "#1" }) int offset, @IndexOrHigh({ "#1" }) int len);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked boolean b);

    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked char c);

    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked int i);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer appendCodePoint(int codePoint);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked long lng);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked float f);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked double d);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer delete(int start, int end);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer deleteCharAt(int index);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer replace(int start, int end, String str);

    @Positive
    @Override
    @Positive
    public synchronized String substring(int start);

    @Positive
    @Override
    @Positive
    public synchronized CharSequence subSequence(int start, int end);

    @Positive
    @Override
    @Positive
    public synchronized String substring(int start, int end);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer insert(int index, char[] str, @IndexOrHigh({ "#2" }) int offset, @IndexOrHigh({ "#2" }) int len);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer insert(int offset, @Nullable Object obj);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer insert(int offset, @Nullable String str);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer insert(int offset, char[] str);

    @Positive
    @Override
    @Positive
    public StringBuffer insert(int dstOffset, @Nullable CharSequence s);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer insert(int dstOffset, @Nullable CharSequence s, @IndexOrHigh({ "#2" }) int start, @IndexOrHigh({ "#2" }) int end);

    @Positive
    @Override
    @Positive
    public StringBuffer insert(int offset, boolean b);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer insert(int offset, char c);

    @Positive
    @Override
    @Positive
    public StringBuffer insert(int offset, int i);

    @Positive
    @Override
    @Positive
    public StringBuffer insert(int offset, long l);

    @Positive
    @Override
    @Positive
    public StringBuffer insert(int offset, float f);

    @Positive
    @Override
    @Positive
    public StringBuffer insert(int offset, double d);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied StringBuffer this, String str);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public synchronized int indexOf(@GuardSatisfied StringBuffer this, String str, int fromIndex);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied StringBuffer this, String str);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public synchronized int lastIndexOf(@GuardSatisfied StringBuffer this, String str, int fromIndex);

    @Positive
    @Override
    @Positive
    public synchronized StringBuffer reverse();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    public synchronized String toString(@GuardSatisfied StringBuffer this);

    @Positive
    synchronized void getBytes(byte[] dst, int dstBegin, byte coder);
    @Positive
}

// CFWR semantic augmentation - variant 1
