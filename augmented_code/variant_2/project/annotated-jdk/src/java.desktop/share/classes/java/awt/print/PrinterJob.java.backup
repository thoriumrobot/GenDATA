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
package java.awt.print;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.AWTError;
import java.awt.HeadlessException;
import javax.print.DocFlavor;
import javax.print.PrintService;
import javax.print.PrintServiceLookup;
import javax.print.StreamPrintServiceFactory;
import javax.print.attribute.AttributeSet;
import javax.print.attribute.PrintRequestAttributeSet;
import javax.print.attribute.standard.Media;
import javax.print.attribute.standard.MediaPrintableArea;
import javax.print.attribute.standard.MediaSize;
import javax.print.attribute.standard.MediaSizeName;
import javax.print.attribute.standard.OrientationRequested;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class PrinterJob {

    public static PrinterJob getPrinterJob();

    public static PrintService[] lookupPrintServices();

    public static StreamPrintServiceFactory[] lookupStreamPrintServices(String mimeType);

    public PrinterJob() {
    }

    public PrintService getPrintService();

    public void setPrintService(PrintService service) throws PrinterException;

    public abstract void setPrintable(Printable painter);

    public abstract void setPrintable(Printable painter, PageFormat format);

    public abstract void setPageable(Pageable document) throws NullPointerException;

    public abstract boolean printDialog() throws HeadlessException;

    public boolean printDialog(PrintRequestAttributeSet attributes) throws HeadlessException;

    public abstract PageFormat pageDialog(PageFormat page) throws HeadlessException;

    public PageFormat pageDialog(PrintRequestAttributeSet attributes) throws HeadlessException;

    public abstract PageFormat defaultPage(PageFormat page);

    public PageFormat defaultPage();

    public PageFormat getPageFormat(PrintRequestAttributeSet attributes);

    public abstract PageFormat validatePage(PageFormat page);

    public abstract void print() throws PrinterException;

    public void print(PrintRequestAttributeSet attributes) throws PrinterException;

    public abstract void setCopies(int copies);

    public abstract int getCopies();

    public abstract String getUserName();

    public abstract void setJobName(String jobName);

    public abstract String getJobName();

    public abstract void cancel();

    public abstract boolean isCancelled();
}
