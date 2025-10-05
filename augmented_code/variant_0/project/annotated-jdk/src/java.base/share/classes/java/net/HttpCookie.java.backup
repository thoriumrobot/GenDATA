/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.List;
import java.util.StringTokenizer;
import java.util.NoSuchElementException;
import java.text.SimpleDateFormat;
import java.util.TimeZone;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.Objects;
import jdk.internal.access.JavaNetHttpCookieAccess;
import jdk.internal.access.SharedSecrets;

public final class HttpCookie implements Cloneable {

    public HttpCookie(String name, String value) {
    }

    public static List<HttpCookie> parse(String header);

    public boolean hasExpired();

    public void setComment(String purpose);

    public String getComment();

    public void setCommentURL(String purpose);

    public String getCommentURL();

    public void setDiscard(boolean discard);

    public boolean getDiscard();

    public void setPortlist(String ports);

    public String getPortlist();

    public void setDomain(String pattern);

    public String getDomain();

    public void setMaxAge(long expiry);

    public long getMaxAge();

    public void setPath(String uri);

    public String getPath();

    public void setSecure(boolean flag);

    public boolean getSecure();

    public String getName();

    public void setValue(String newValue);

    public String getValue();

    public int getVersion();

    public void setVersion(int v);

    public boolean isHttpOnly();

    public void setHttpOnly(boolean httpOnly);

    public static boolean domainMatches(String domain, String host);

    @Override
    public String toString();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public Object clone();

    long getCreationTime();

    static interface CookieAttributeAssignor {

        public void assign(HttpCookie cookie, String attrName, String attrValue);
    }
}
