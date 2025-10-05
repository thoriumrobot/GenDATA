/*
 * Copyright (c) 1994, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.common.aliasing.qual.Unique;
import org.checkerframework.common.aliasing.qual.LeakedToResult;
import org.checkerframework.common.aliasing.qual.NonLeaked;
import java.io.IOException;
import java.util.Arrays;
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "aliasing", "lock", "nullness", "index" })
public final class StringBuffer extends AbstractStringBuilder implements java.io.Serializable, Comparable<StringBuffer>, CharSequence {

    @IntrinsicCandidate
    @Unique
    public StringBuffer() {
    }

    @IntrinsicCandidate
    @Unique
    public StringBuffer(@NonNegative int capacity) {
    }

    @IntrinsicCandidate
    @Unique
    public StringBuffer(String str) {
    }

    @Unique
    public StringBuffer(CharSequence seq) {
    }

    @Override
    public synchronized int compareTo(StringBuffer another);

    @Pure
    @Override
    @NonNegative
    public synchronized int length(@GuardSatisfied StringBuffer this);

    @Override
    @NonNegative
    public synchronized int capacity();

    @Override
    public synchronized void ensureCapacity(int minimumCapacity);

    @Override
    public synchronized void trimToSize();

    @Override
    public synchronized void setLength(@NonNegative int newLength);

    @Override
    public synchronized char charAt(int index);

    @Override
    public synchronized int codePointAt(int index);

    @Override
    public synchronized int codePointBefore(int index);

    @Override
    public synchronized int codePointCount(int beginIndex, int endIndex);

    @Override
    public synchronized int offsetByCodePoints(int index, int codePointOffset);

    @Override
    public synchronized void getChars(int srcBegin, int srcEnd, char[] dst, @IndexOrHigh({ "#3" }) int dstBegin);

    @Override
    public synchronized void setCharAt(int index, char ch);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable Object obj);

    @Override
    @IntrinsicCandidate
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable String str);

    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable StringBuffer sb);

    @Override
    synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked AbstractStringBuilder asb);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable CharSequence s);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked @Nullable CharSequence s, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked char[] str);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked char[] str, @IndexOrHigh({ "#1" }) int offset, @IndexOrHigh({ "#1" }) int len);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked boolean b);

    @Override
    @IntrinsicCandidate
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked char c);

    @Override
    @IntrinsicCandidate
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked int i);

    @Override
    public synchronized StringBuffer appendCodePoint(int codePoint);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked long lng);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked float f);

    @Override
    public synchronized StringBuffer append(@LeakedToResult StringBuffer this, @NonLeaked double d);

    @Override
    public synchronized StringBuffer delete(int start, int end);

    @Override
    public synchronized StringBuffer deleteCharAt(int index);

    @Override
    public synchronized StringBuffer replace(int start, int end, String str);

    @Override
    public synchronized String substring(int start);

    @Override
    public synchronized CharSequence subSequence(int start, int end);

    @Override
    public synchronized String substring(int start, int end);

    @Override
    public synchronized StringBuffer insert(int index, char[] str, @IndexOrHigh({ "#2" }) int offset, @IndexOrHigh({ "#2" }) int len);

    @Override
    public synchronized StringBuffer insert(int offset, @Nullable Object obj);

    @Override
    public synchronized StringBuffer insert(int offset, @Nullable String str);

    @Override
    public synchronized StringBuffer insert(int offset, char[] str);

    @Override
    public StringBuffer insert(int dstOffset, @Nullable CharSequence s);

    @Override
    public synchronized StringBuffer insert(int dstOffset, @Nullable CharSequence s, @IndexOrHigh({ "#2" }) int start, @IndexOrHigh({ "#2" }) int end);

    @Override
    public StringBuffer insert(int offset, boolean b);

    @Override
    public synchronized StringBuffer insert(int offset, char c);

    @Override
    public StringBuffer insert(int offset, int i);

    @Override
    public StringBuffer insert(int offset, long l);

    @Override
    public StringBuffer insert(int offset, float f);

    @Override
    public StringBuffer insert(int offset, double d);

    @Pure
    @Override
    @GTENegativeOne
    public int indexOf(@GuardSatisfied StringBuffer this, String str);

    @Pure
    @Override
    @GTENegativeOne
    public synchronized int indexOf(@GuardSatisfied StringBuffer this, String str, int fromIndex);

    @Pure
    @Override
    @GTENegativeOne
    public int lastIndexOf(@GuardSatisfied StringBuffer this, String str);

    @Pure
    @Override
    @GTENegativeOne
    public synchronized int lastIndexOf(@GuardSatisfied StringBuffer this, String str, int fromIndex);

    @Override
    public synchronized StringBuffer reverse();

    @SideEffectFree
    @Override
    @IntrinsicCandidate
    public synchronized String toString(@GuardSatisfied StringBuffer this);

    synchronized void getBytes(byte[] dst, int dstBegin, byte coder);
}
