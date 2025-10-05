/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.print;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.AWTError;
    @Positive
import java.awt.HeadlessException;
    @Positive
import javax.print.DocFlavor;
    @Positive
import javax.print.PrintService;
    @Positive
import javax.print.PrintServiceLookup;
    @Positive
import javax.print.StreamPrintServiceFactory;
    @Positive
import javax.print.attribute.AttributeSet;
    @Positive
import javax.print.attribute.PrintRequestAttributeSet;
    @Positive
import javax.print.attribute.standard.Media;
    @Positive
import javax.print.attribute.standard.MediaPrintableArea;
    @Positive
import javax.print.attribute.standard.MediaSize;
    @Positive
import javax.print.attribute.standard.MediaSizeName;
    @Positive
import javax.print.attribute.standard.OrientationRequested;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class PrinterJob {

    @Positive
    public static PrinterJob getPrinterJob();

    @Positive
    public static PrintService[] lookupPrintServices();

    @Positive
    public static StreamPrintServiceFactory[] lookupStreamPrintServices(String mimeType);

    @Positive
    public PrinterJob() {
    @Positive
    }

    @Positive
    public PrintService getPrintService();

    @Positive
    public void setPrintService(PrintService service) throws PrinterException;

    @Positive
    public abstract void setPrintable(Printable painter);

    @Positive
    public abstract void setPrintable(Printable painter, PageFormat format);

    @Positive
    public abstract void setPageable(Pageable document) throws NullPointerException;

    @Positive
    public abstract boolean printDialog() throws HeadlessException;

    @Positive
    public boolean printDialog(PrintRequestAttributeSet attributes) throws HeadlessException;

    @Positive
    public abstract PageFormat pageDialog(PageFormat page) throws HeadlessException;

    @Positive
    public PageFormat pageDialog(PrintRequestAttributeSet attributes) throws HeadlessException;

    @Positive
    public abstract PageFormat defaultPage(PageFormat page);

    @Positive
    public PageFormat defaultPage();

    @Positive
    public PageFormat getPageFormat(PrintRequestAttributeSet attributes);

    @Positive
    public abstract PageFormat validatePage(PageFormat page);

    @Positive
    public abstract void print() throws PrinterException;

    @Positive
    public void print(PrintRequestAttributeSet attributes) throws PrinterException;

    @Positive
    public abstract void setCopies(int copies);

    @Positive
    public abstract int getCopies();

    @Positive
    public abstract String getUserName();

    @Positive
    public abstract void setJobName(String jobName);

    @Positive
    public abstract String getJobName();

    @Positive
    public abstract void cancel();

    @Positive
    public abstract boolean isCancelled();
    @Positive
}

// CFWR semantic augmentation - variant 0
