/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.signedness.qual.SignednessGlb;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.Character;
import java.util.Vector;
import sun.text.CollatorUtilities;
import jdk.internal.icu.text.NormalizerBase;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class CollationElementIterator {

    @SignednessGlb
    public static final int NULLORDER;

    public void reset();

    public int next();

    public int previous();

    public static final int primaryOrder(int order);

    public static final short secondaryOrder(int order);

    public static final short tertiaryOrder(int order);

    final int strengthOrder(int order);

    @SuppressWarnings("deprecation")
    public void setOffset(int newOffset);

    public int getOffset();

    public int getMaxExpansion(int order);

    public void setText(String source);

    public void setText(CharacterIterator source);

    static final boolean isIgnorable(int order);
}
