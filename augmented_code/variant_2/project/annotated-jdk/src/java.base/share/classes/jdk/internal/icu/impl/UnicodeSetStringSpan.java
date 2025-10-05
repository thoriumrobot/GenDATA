/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.icu.impl;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import jdk.internal.icu.text.UTF16;
    @Positive
import jdk.internal.icu.text.UnicodeSet;
    @Positive
import jdk.internal.icu.text.UnicodeSet.SpanCondition;
    @Positive
import jdk.internal.icu.util.OutputInt;

    @Positive
public class UnicodeSetStringSpan {

    @Positive
    public static final int WITH_COUNT;

    @Positive
    public static final int FWD;

    @Positive
    public static final int BACK;

    @Positive
    public static final int CONTAINED;

    @Positive
    public static final int NOT_CONTAINED;

    @Positive
    public static final int ALL;

    @Positive
    public static final int FWD_UTF16_CONTAINED;

    @Positive
    public static final int FWD_UTF16_NOT_CONTAINED;

    @Positive
    public static final int BACK_UTF16_CONTAINED;

    @Positive
    public static final int BACK_UTF16_NOT_CONTAINED;

    @Positive
    public UnicodeSetStringSpan(final UnicodeSet set, final ArrayList<String> setStrings, int which) {
    @Positive
    }

    @Positive
    public boolean needsStringSpanUTF16();

    @Positive
    @Pure
    @Positive
    public boolean contains(int c);

    @Positive
    public int span(CharSequence s, int start, SpanCondition spanCondition);

    @Positive
    public int spanAndCount(CharSequence s, int start, SpanCondition spanCondition, OutputInt outCount);

    @Positive
    public synchronized int spanBack(CharSequence s, int length, SpanCondition spanCondition);

    @Positive
    static short makeSpanLengthByte(int spanLength);

    @Positive
    static boolean matches16CPB(CharSequence s, int start, int limit, final String t, int tlength);

    @Positive
    static int spanOne(final UnicodeSet set, CharSequence s, int start, int length);

    @Positive
    static int spanOneBack(final UnicodeSet set, CharSequence s, int length);

    @Positive
    private static final class OffsetList {

    @Positive
        public OffsetList() {
    @Positive
        }

    @Positive
        public void setMaxLength(int maxLength);

    @Positive
        public void clear();

    @Positive
        public boolean isEmpty();

    @Positive
        public void shift(int delta);

    @Positive
        public void addOffset(int offset);

    @Positive
        public void addOffsetAndCount(int offset, int count);

    @Positive
        @Pure
    @Positive
        public boolean containsOffset(int offset);

    @Positive
        public boolean hasCountAtOffset(int offset, int count);

    @Positive
        public int popMinimum(OutputInt outCount);
    @Positive
    }
    @Positive
}
