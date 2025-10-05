/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.datatransfer;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import sun.datatransfer.DataFlavorUtil;
    @Positive
import sun.datatransfer.DesktopDatatransferService;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class SystemFlavorMap implements FlavorMap, FlavorTable {

    @Positive
    public static FlavorMap getDefaultFlavorMap();

    @Positive
    @Override
    @Positive
    public synchronized List<String> getNativesForFlavor(DataFlavor flav);

    @Positive
    @Override
    @Positive
    public synchronized List<DataFlavor> getFlavorsForNative(String nat);

    @Positive
    @Override
    @Positive
    public synchronized Map<DataFlavor, String> getNativesForFlavors(DataFlavor[] flavors);

    @Positive
    @Override
    @Positive
    public synchronized Map<String, DataFlavor> getFlavorsForNatives(String[] natives);

    @Positive
    public synchronized void addUnencodedNativeForFlavor(DataFlavor flav, String nat);

    @Positive
    public synchronized void setNativesForFlavor(DataFlavor flav, String[] natives);

    @Positive
    public synchronized void addFlavorForUnencodedNative(String nat, DataFlavor flav);

    @Positive
    public synchronized void setFlavorsForNative(String nat, DataFlavor[] flavors);

    @Positive
    public static String encodeJavaMIMEType(String mimeType);

    @Positive
    public static String encodeDataFlavor(DataFlavor flav);

    @Positive
    public static boolean isJavaMIMEType(String str);

    @Positive
    public static String decodeJavaMIMEType(String nat);

    @Positive
    public static DataFlavor decodeDataFlavor(String nat) throws ClassNotFoundException;

    @Positive
    private static final class SoftCache<K, V> {

    @Positive
        public void put(K key, LinkedHashSet<V> value);

    @Positive
        public void remove(K key);

    @Positive
        public LinkedHashSet<V> check(K key);
    @Positive
    }
    @Positive
}
