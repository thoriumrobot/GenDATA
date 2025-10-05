/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.net.www;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.io.*;
    @Positive
import java.util.Collections;
    @Positive
import java.util.*;

    @Positive
public class MessageHeader {

    @Positive
    public MessageHeader() {
    @Positive
    }

    @Positive
    public MessageHeader(InputStream is) throws java.io.IOException {
    @Positive
    }

    @Positive
    public synchronized String getHeaderNamesInList();

    @Positive
    public synchronized void reset();

    @Positive
    public synchronized String findValue(String k);

    @Positive
    public synchronized int getKey(String k);

    @Positive
    public synchronized String getKey(int n);

    @Positive
    public synchronized String getValue(int n);

    @Positive
    public synchronized String findNextValue(String k, String v);

    @Positive
    public boolean filterNTLMResponses(String k);

    @Positive
    class HeaderIterator implements Iterator<String> {

    @Positive
        public HeaderIterator(String k, Object lock) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public String next();

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    public Iterator<String> multiValueIterator(String k);

    @Positive
    public synchronized Map<String, List<String>> getHeaders();

    @Positive
    public synchronized Map<String, List<String>> getHeaders(String[] excludeList);

    @Positive
    public synchronized Map<String, List<String>> filterAndAddHeaders(String[] excludeList, Map<String, List<String>> include);

    @Positive
    public void print(PrintStream p);

    @Positive
    public synchronized void add(String k, String v);

    @Positive
    public synchronized void prepend(String k, String v);

    @Positive
    public synchronized void set(int i, String k, String v);

    @Positive
    public synchronized void remove(String k);

    @Positive
    public synchronized void set(String k, String v);

    @Positive
    public synchronized void setIfNotSet(String k, String v);

    @Positive
    public static String canonicalID(String id);

    @Positive
    public void parseHeader(InputStream is) throws java.io.IOException;

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public void mergeHeader(InputStream is) throws java.io.IOException;

    @Positive
    public synchronized String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
