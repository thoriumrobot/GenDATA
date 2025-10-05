/*
 * Copyright (c) 2006, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.framework.qual.AnnotatedFor;

@AnnotatedFor({ "index", "interning" })
@UsesObjectEquals
abstract class CharacterData {

    abstract int getProperties(int ch);

    abstract int getType(int ch);

    abstract boolean isDigit(int ch);

    abstract boolean isLowerCase(int ch);

    abstract boolean isUpperCase(int ch);

    abstract boolean isWhitespace(int ch);

    abstract boolean isMirrored(int ch);

    abstract boolean isJavaIdentifierStart(int ch);

    abstract boolean isJavaIdentifierPart(int ch);

    abstract boolean isUnicodeIdentifierStart(int ch);

    abstract boolean isUnicodeIdentifierPart(int ch);

    abstract boolean isIdentifierIgnorable(int ch);

    abstract int toLowerCase(int ch);

    abstract int toUpperCase(int ch);

    abstract int toTitleCase(int ch);

    abstract int digit(int ch, @IntRange(from = 2, to = 36) int radix);

    abstract int getNumericValue(int ch);

    abstract byte getDirectionality(int ch);

    int toUpperCaseEx(int ch);

    char[] toUpperCaseCharArray(int ch);

    boolean isOtherAlphabetic(int ch);

    boolean isIdeographic(int ch);

    static final CharacterData of(int ch);
}
