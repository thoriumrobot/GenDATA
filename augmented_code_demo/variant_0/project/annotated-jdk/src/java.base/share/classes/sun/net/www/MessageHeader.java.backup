/*
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.net.www;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.io.*;
import java.util.Collections;
import java.util.*;

public class MessageHeader {

    public MessageHeader() {
    }

    public MessageHeader(InputStream is) throws java.io.IOException {
    }

    public synchronized String getHeaderNamesInList();

    public synchronized void reset();

    public synchronized String findValue(String k);

    public synchronized int getKey(String k);

    public synchronized String getKey(int n);

    public synchronized String getValue(int n);

    public synchronized String findNextValue(String k, String v);

    public boolean filterNTLMResponses(String k);

    class HeaderIterator implements Iterator<String> {

        public HeaderIterator(String k, Object lock) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public String next();

        public void remove();
    }

    public Iterator<String> multiValueIterator(String k);

    public synchronized Map<String, List<String>> getHeaders();

    public synchronized Map<String, List<String>> getHeaders(String[] excludeList);

    public synchronized Map<String, List<String>> filterAndAddHeaders(String[] excludeList, Map<String, List<String>> include);

    public void print(PrintStream p);

    public synchronized void add(String k, String v);

    public synchronized void prepend(String k, String v);

    public synchronized void set(int i, String k, String v);

    public synchronized void remove(String k);

    public synchronized void set(String k, String v);

    public synchronized void setIfNotSet(String k, String v);

    public static String canonicalID(String id);

    public void parseHeader(InputStream is) throws java.io.IOException;

    @SuppressWarnings("fallthrough")
    public void mergeHeader(InputStream is) throws java.io.IOException;

    public synchronized String toString();
}
