/*
 * Copyright (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Vector;
import sun.text.UCompactIntArray;
import sun.text.IntHashtable;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
final class RBCollationTables {

    public RBCollationTables(String rules, int decmp) throws ParseException {
    }

    final class BuildAPI {

        void fillInTables(boolean f2ary, boolean swap, UCompactIntArray map, Vector<Vector<EntryPair>> cTbl, Vector<int[]> eTbl, IntHashtable cFlgs, short mso, short mto);
    }

    public String getRules();

    public boolean isFrenchSec();

    public boolean isSEAsianSwapping();

    Vector<EntryPair> getContractValues(int ch);

    boolean usedInContractSeq(int c);

    int getMaxExpansion(int order);

    final int[] getExpandValueList(int idx);

    int getUnicodeOrder(int ch);

    short getMaxSecOrder();

    short getMaxTerOrder();

    static void reverse(StringBuffer result, int from, int to);

    static final int getEntry(Vector<EntryPair> list, String name, boolean fwd);
}
