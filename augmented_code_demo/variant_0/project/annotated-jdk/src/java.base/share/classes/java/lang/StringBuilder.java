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
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.regex.qual.PolyRegex;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import java.io.IOException;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index", "regex", "sideeffectsonly" })
    @Positive
public final class StringBuilder extends AbstractStringBuilder implements java.io.Serializable, Comparable<StringBuilder>, CharSequence {

    @Positive
    @IntrinsicCandidate
    @Positive
    public StringBuilder() {
    @Positive
    }

    @Positive
    @IntrinsicCandidate
    @Positive
    public StringBuilder(@NonNegative int capacity) {
    @Positive
    }

    @Positive
    @IntrinsicCandidate
    @Positive
    public StringBuilder(String str) {
    @Positive
    }

    @Positive
    public StringBuilder(CharSequence seq) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public int compareTo(StringBuilder another);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(@GuardSatisfied @Nullable Object obj);

    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(@Nullable String str);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(@Nullable StringBuffer sb);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(@Nullable CharSequence s);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(@Nullable CharSequence s, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(char[] str);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(char[] str, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(boolean b);

    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(char c);

    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(int i);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(long lng);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(float f);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder append(double d);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder appendCodePoint(int codePoint);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder delete(@NonNegative int start, @NonNegative int end);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder deleteCharAt(@NonNegative int index);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder replace(@NonNegative int start, @NonNegative int end, String str);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int index, char[] str, @IndexOrHigh({ "#2" }) int offset, @IndexOrHigh({ "#2" }) int len);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, @Nullable String str);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, char[] str);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int dstOffset, @Nullable CharSequence s);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int dstOffset, @Nullable CharSequence s, @NonNegative int start, @NonNegative int end);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, boolean b);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, char c);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, int i);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, long l);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, float f);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder insert(@NonNegative int offset, double d);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied StringBuilder this, String str);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied StringBuilder this, String str, int fromIndex);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied StringBuilder this, String str);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied StringBuilder this, String str, int fromIndex);

    @Positive
    @Override
    @Positive
    @SideEffectsOnly("this")
    @Positive
    public StringBuilder reverse();

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyRegex
    @Positive
    public String toString(@GuardSatisfied @PolyRegex StringBuilder this);
    @Positive
}

// CFWR semantic augmentation - variant 0
