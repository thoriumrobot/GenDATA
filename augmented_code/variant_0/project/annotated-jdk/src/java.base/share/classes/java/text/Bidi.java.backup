/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.icu.text.BidiBase;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class Bidi {

    public static final int DIRECTION_LEFT_TO_RIGHT;

    public static final int DIRECTION_RIGHT_TO_LEFT;

    public static final int DIRECTION_DEFAULT_LEFT_TO_RIGHT;

    public static final int DIRECTION_DEFAULT_RIGHT_TO_LEFT;

    public Bidi(String paragraph, int flags) {
    }

    public Bidi(AttributedCharacterIterator paragraph) {
    }

    public Bidi(char[] text, int textStart, byte[] embeddings, int embStart, int paragraphLength, int flags) {
    }

    public Bidi createLineBidi(int lineStart, int lineLimit);

    public boolean isMixed();

    public boolean isLeftToRight();

    public boolean isRightToLeft();

    public int getLength();

    public boolean baseIsLeftToRight();

    public int getBaseLevel();

    public int getLevelAt(int offset);

    public int getRunCount();

    public int getRunLevel(int run);

    public int getRunStart(int run);

    public int getRunLimit(int run);

    public static boolean requiresBidi(char[] text, int start, int limit);

    public static void reorderVisually(byte[] levels, int levelStart, Object[] objects, int objectStart, int count);

    public String toString();
}
