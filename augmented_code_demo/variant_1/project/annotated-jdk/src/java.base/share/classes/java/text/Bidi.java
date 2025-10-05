/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.icu.text.BidiBase;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Bidi {

    @Positive
    public static final int DIRECTION_LEFT_TO_RIGHT;

    @Positive
    public static final int DIRECTION_RIGHT_TO_LEFT;

    @Positive
    public static final int DIRECTION_DEFAULT_LEFT_TO_RIGHT;

    @Positive
    public static final int DIRECTION_DEFAULT_RIGHT_TO_LEFT;

    @Positive
    public Bidi(String paragraph, int flags) {
    @Positive
    }

    @Positive
    public Bidi(AttributedCharacterIterator paragraph) {
    @Positive
    }

    @Positive
    public Bidi(char[] text, int textStart, byte[] embeddings, int embStart, int paragraphLength, int flags) {
    @Positive
    }

    @Positive
    public Bidi createLineBidi(int lineStart, int lineLimit);

    @Positive
    public boolean isMixed();

    @Positive
    public boolean isLeftToRight();

    @Positive
    public boolean isRightToLeft();

    @Positive
    public int getLength();

    @Positive
    public boolean baseIsLeftToRight();

    @Positive
    public int getBaseLevel();

    @Positive
    public int getLevelAt(int offset);

    @Positive
    public int getRunCount();

    @Positive
    public int getRunLevel(int run);

    @Positive
    public int getRunStart(int run);

    @Positive
    public int getRunLimit(int run);

    @Positive
    public static boolean requiresBidi(char[] text, int start, int limit);

    @Positive
    public static void reorderVisually(byte[] levels, int levelStart, Object[] objects, int objectStart, int count);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
