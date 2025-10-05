/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.text;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Vector;
    @Positive
import sun.text.UCompactIntArray;
    @Positive
import sun.text.IntHashtable;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
final class RBCollationTables {

    @Positive
    public RBCollationTables(String rules, int decmp) throws ParseException {
    @Positive
    }

    @Positive
    final class BuildAPI {

    @Positive
        void fillInTables(boolean f2ary, boolean swap, UCompactIntArray map, Vector<Vector<EntryPair>> cTbl, Vector<int[]> eTbl, IntHashtable cFlgs, short mso, short mto);
    @Positive
    }

    @Positive
    public String getRules();

    @Positive
    public boolean isFrenchSec();

    @Positive
    public boolean isSEAsianSwapping();

    @Positive
    Vector<EntryPair> getContractValues(int ch);

    @Positive
    boolean usedInContractSeq(int c);

    @Positive
    int getMaxExpansion(int order);

    @Positive
    final int[] getExpandValueList(int idx);

    @Positive
    int getUnicodeOrder(int ch);

    @Positive
    short getMaxSecOrder();

    @Positive
    short getMaxTerOrder();

    @Positive
    static void reverse(StringBuffer result, int from, int to);

    @Positive
    static final int getEntry(Vector<EntryPair> list, String name, boolean fwd);
    @Positive
}
