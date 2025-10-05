/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.datatransfer;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.ref.SoftReference;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import sun.datatransfer.DataFlavorUtil;
import sun.datatransfer.DesktopDatatransferService;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public final class SystemFlavorMap implements FlavorMap, FlavorTable {

    public static FlavorMap getDefaultFlavorMap();

    @Override
    public synchronized List<String> getNativesForFlavor(DataFlavor flav);

    @Override
    public synchronized List<DataFlavor> getFlavorsForNative(String nat);

    @Override
    public synchronized Map<DataFlavor, String> getNativesForFlavors(DataFlavor[] flavors);

    @Override
    public synchronized Map<String, DataFlavor> getFlavorsForNatives(String[] natives);

    public synchronized void addUnencodedNativeForFlavor(DataFlavor flav, String nat);

    public synchronized void setNativesForFlavor(DataFlavor flav, String[] natives);

    public synchronized void addFlavorForUnencodedNative(String nat, DataFlavor flav);

    public synchronized void setFlavorsForNative(String nat, DataFlavor[] flavors);

    public static String encodeJavaMIMEType(String mimeType);

    public static String encodeDataFlavor(DataFlavor flav);

    public static boolean isJavaMIMEType(String str);

    public static String decodeJavaMIMEType(String nat);

    public static DataFlavor decodeDataFlavor(String nat) throws ClassNotFoundException;

    private static final class SoftCache<K, V> {

        public void put(K key, LinkedHashSet<V> value);

        public void remove(K key);

        public LinkedHashSet<V> check(K key);
    }
}
