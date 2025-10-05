/*
    @Positive
 * Copyright (c) 2005, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.icu.text;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.text.ParsePosition;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.TreeSet;
    @Positive
import jdk.internal.icu.impl.BMPSet;
    @Positive
import jdk.internal.icu.impl.UCharacterProperty;
    @Positive
import jdk.internal.icu.impl.UnicodeSetStringSpan;
    @Positive
import jdk.internal.icu.impl.Utility;
    @Positive
import jdk.internal.icu.lang.UCharacter;
    @Positive
import jdk.internal.icu.util.OutputInt;
    @Positive
import jdk.internal.icu.util.VersionInfo;

    @Positive
public class UnicodeSet {

    @Positive
    public static final int MIN_VALUE;

    @Positive
    public static final int MAX_VALUE;

    @Positive
    public UnicodeSet(int start, int end) {
    @Positive
    }

    @Positive
    public UnicodeSet(String pattern) {
    @Positive
    }

    @Positive
    public UnicodeSet set(UnicodeSet other);

    @Positive
    public int size();

    @Positive
    public final UnicodeSet add(int c);

    @Positive
    public final UnicodeSet add(CharSequence s);

    @Positive
    public UnicodeSet complement(int start, int end);

    @Positive
    @Pure
    @Positive
    public boolean contains(int c);

    @Positive
    public UnicodeSet retainAll(UnicodeSet c);

    @Positive
    public UnicodeSet clear();

    @Positive
    public int getRangeCount();

    @Positive
    public int getRangeStart(int index);

    @Positive
    public int getRangeEnd(int index);

    @Positive
    private static interface Filter {

    @Positive
        @Pure
    @Positive
        boolean contains(int codePoint);
    @Positive
    }

    @Positive
    private static class VersionFilter implements Filter {

    @Positive
        @Pure
    @Positive
        public boolean contains(int ch);
    @Positive
    }

    @Positive
    public boolean isFrozen();

    @Positive
    public UnicodeSet freeze();

    @Positive
    public int span(CharSequence s, SpanCondition spanCondition);

    @Positive
    public int span(CharSequence s, int start, SpanCondition spanCondition);

    @Positive
    public int spanAndCount(CharSequence s, int start, SpanCondition spanCondition, OutputInt outCount);

    @Positive
    public int spanBack(CharSequence s, int fromIndex, SpanCondition spanCondition);

    @Positive
    public UnicodeSet cloneAsThawed();

    @Positive
    public enum SpanCondition {

    @Positive
        NOT_CONTAINED, CONTAINED, SIMPLE
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
