/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.datatype;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import javax.xml.namespace.QName;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.TimeZone;
import java.util.GregorianCalendar;

public abstract class XMLGregorianCalendar implements Cloneable {

    public XMLGregorianCalendar() {
    }

    public abstract void clear();

    public abstract void reset();

    public abstract void setYear(BigInteger year);

    public abstract void setYear(int year);

    public abstract void setMonth(int month);

    public abstract void setDay(int day);

    public abstract void setTimezone(int offset);

    public void setTime(int hour, int minute, int second);

    public abstract void setHour(int hour);

    public abstract void setMinute(int minute);

    public abstract void setSecond(int second);

    public abstract void setMillisecond(int millisecond);

    public abstract void setFractionalSecond(BigDecimal fractional);

    public void setTime(int hour, int minute, int second, BigDecimal fractional);

    public void setTime(int hour, int minute, int second, int millisecond);

    public abstract BigInteger getEon();

    public abstract int getYear();

    public abstract BigInteger getEonAndYear();

    public abstract int getMonth();

    public abstract int getDay();

    public abstract int getTimezone();

    public abstract int getHour();

    public abstract int getMinute();

    public abstract int getSecond();

    public int getMillisecond();

    public abstract BigDecimal getFractionalSecond();

    public abstract int compare(XMLGregorianCalendar xmlGregorianCalendar);

    public abstract XMLGregorianCalendar normalize();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    public abstract String toXMLFormat();

    public abstract QName getXMLSchemaType();

    @Override
    public String toString();

    public abstract boolean isValid();

    public abstract void add(Duration duration);

    public abstract GregorianCalendar toGregorianCalendar();

    public abstract GregorianCalendar toGregorianCalendar(java.util.TimeZone timezone, java.util.Locale aLocale, XMLGregorianCalendar defaults);

    public abstract TimeZone getTimeZone(int defaultZoneoffset);

    @Override
    public abstract Object clone();
}
