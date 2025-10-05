/*
 * Copyright (c) 2004, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.org.apache.xerces.internal.jaxp.datatype;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.xerces.internal.util.DatatypeMessageFormatter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.TimeZone;
import javax.xml.datatype.DatatypeConstants;
import javax.xml.datatype.Duration;
import javax.xml.datatype.XMLGregorianCalendar;
import javax.xml.namespace.QName;
import jdk.xml.internal.SecuritySupport;

public class XMLGregorianCalendarImpl extends XMLGregorianCalendar implements Serializable, Cloneable {

    public static final XMLGregorianCalendar LEAP_YEAR_DEFAULT;

    protected XMLGregorianCalendarImpl(String lexicalRepresentation) throws IllegalArgumentException {
    }

    public XMLGregorianCalendarImpl() {
    }

    protected XMLGregorianCalendarImpl(BigInteger year, int month, int day, int hour, int minute, int second, BigDecimal fractionalSecond, int timezone) {
    }

    public XMLGregorianCalendarImpl(GregorianCalendar cal) {
    }

    public static XMLGregorianCalendar createDateTime(BigInteger year, int month, int day, int hours, int minutes, int seconds, BigDecimal fractionalSecond, int timezone);

    public static XMLGregorianCalendar createDateTime(int year, int month, int day, int hour, int minute, int second);

    public static XMLGregorianCalendar createDateTime(int year, int month, int day, int hours, int minutes, int seconds, int milliseconds, int timezone);

    public static XMLGregorianCalendar createDate(int year, int month, int day, int timezone);

    public static XMLGregorianCalendar createTime(int hours, int minutes, int seconds, int timezone);

    public static XMLGregorianCalendar createTime(int hours, int minutes, int seconds, BigDecimal fractionalSecond, int timezone);

    public static XMLGregorianCalendar createTime(int hours, int minutes, int seconds, int milliseconds, int timezone);

    public BigInteger getEon();

    public int getYear();

    public BigInteger getEonAndYear();

    public int getMonth();

    public int getDay();

    public int getTimezone();

    public int getHour();

    public int getMinute();

    public int getSecond();

    public int getMillisecond();

    public BigDecimal getFractionalSecond();

    public final void setYear(BigInteger year);

    public final void setYear(int year);

    public final void setMonth(int month);

    public final void setDay(int day);

    public final void setTimezone(int offset);

    public final void setTime(int hour, int minute, int second);

    public void setHour(int hour);

    public void setMinute(int minute);

    public void setSecond(int second);

    public final void setTime(int hour, int minute, int second, BigDecimal fractional);

    public final void setTime(int hour, int minute, int second, int millisecond);

    public int compare(XMLGregorianCalendar rhs);

    public XMLGregorianCalendar normalize();

    public static XMLGregorianCalendar parse(String lexicalRepresentation);

    public String toXMLFormat();

    public QName getXMLSchemaType();

    public final boolean isValid();

    public void add(Duration duration);

    private static class DaysInMonth {
    }

    public java.util.GregorianCalendar toGregorianCalendar();

    public GregorianCalendar toGregorianCalendar(TimeZone timezone, Locale aLocale, XMLGregorianCalendar defaults);

    public TimeZone getTimeZone(int defaultZoneoffset);

    public Object clone();

    public void clear();

    public void setMillisecond(int millisecond);

    public final void setFractionalSecond(BigDecimal fractional);

    private final class Parser {

        public void parse() throws IllegalArgumentException;
    }

    static BigInteger sanitize(Number value, int signum);

    public void reset();
}
