/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.print;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.GraphicsEnvironment;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.print.PrinterJob;
import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import javax.print.DocFlavor;
import javax.print.DocPrintJob;
import javax.print.PrintService;
import javax.print.ServiceUIFactory;
import javax.print.attribute.Attribute;
import javax.print.attribute.AttributeSet;
import javax.print.attribute.AttributeSetUtilities;
import javax.print.attribute.EnumSyntax;
import javax.print.attribute.HashAttributeSet;
import javax.print.attribute.PrintRequestAttributeSet;
import javax.print.attribute.PrintServiceAttribute;
import javax.print.attribute.PrintServiceAttributeSet;
import javax.print.attribute.HashPrintServiceAttributeSet;
import javax.print.attribute.standard.PrinterName;
import javax.print.attribute.standard.PrinterIsAcceptingJobs;
import javax.print.attribute.standard.QueuedJobCount;
import javax.print.attribute.standard.JobName;
import javax.print.attribute.standard.RequestingUserName;
import javax.print.attribute.standard.Chromaticity;
import javax.print.attribute.standard.Copies;
import javax.print.attribute.standard.CopiesSupported;
import javax.print.attribute.standard.Destination;
import javax.print.attribute.standard.DialogOwner;
import javax.print.attribute.standard.DialogTypeSelection;
import javax.print.attribute.standard.Fidelity;
import javax.print.attribute.standard.Media;
import javax.print.attribute.standard.MediaSizeName;
import javax.print.attribute.standard.MediaSize;
import javax.print.attribute.standard.MediaTray;
import javax.print.attribute.standard.MediaPrintableArea;
import javax.print.attribute.standard.OrientationRequested;
import javax.print.attribute.standard.PageRanges;
import javax.print.attribute.standard.PrinterState;
import javax.print.attribute.standard.PrinterStateReason;
import javax.print.attribute.standard.PrinterStateReasons;
import javax.print.attribute.standard.Severity;
import javax.print.attribute.standard.Sides;
import javax.print.attribute.standard.ColorSupported;
import javax.print.attribute.standard.PrintQuality;
import javax.print.attribute.standard.PrinterResolution;
import javax.print.attribute.standard.SheetCollate;
import javax.print.event.PrintServiceAttributeListener;
import sun.awt.windows.WPrinterJob;

public class Win32PrintService implements PrintService, AttributeUpdater, SunPrinterJobService {

    public static MediaSize[] predefMedia;

    public static final MediaSizeName[] dmPaperToPrintService;

    public void invalidateService();

    public String getName();

    public int findPaperID(MediaSizeName msn);

    public int findTrayID(MediaTray tray);

    public MediaTray findMediaTray(int dmBin);

    public MediaSizeName findWin32Media(int dmIndex);

    public MediaSizeName findMatchingMediaSizeNameMM(float w, float h);

    public DocPrintJob createPrintJob();

    public PrintServiceAttributeSet getUpdatedAttributes();

    public void wakeNotifier();

    public void addPrintServiceAttributeListener(PrintServiceAttributeListener listener);

    public void removePrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @SuppressWarnings("unchecked")
    public <T extends PrintServiceAttribute> T getAttribute(Class<T> category);

    public PrintServiceAttributeSet getAttributes();

    public DocFlavor[] getSupportedDocFlavors();

    public boolean isDocFlavorSupported(DocFlavor flavor);

    public Class<?>[] getSupportedAttributeCategories();

    public boolean isAttributeCategorySupported(Class<? extends Attribute> category);

    public Object getDefaultAttributeValue(Class<? extends Attribute> category);

    public Object getSupportedAttributeValues(Class<? extends Attribute> category, DocFlavor flavor, AttributeSet attributes);

    public boolean isAttributeValueSupported(Attribute attr, DocFlavor flavor, AttributeSet attributes);

    public AttributeSet getUnsupportedAttributes(DocFlavor flavor, AttributeSet attributes);

    private static class Win32DocumentPropertiesUI extends DocumentPropertiesUI {

        public PrintRequestAttributeSet showDocumentProperties(PrinterJob job, Window owner, PrintService service, PrintRequestAttributeSet aset);
    }

    private static class Win32ServiceUIFactory extends ServiceUIFactory {

        public Object getUI(int role, String ui);

        public String[] getUIClassNamesForRole(int role);
    }

    public synchronized ServiceUIFactory getServiceUIFactory();

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public boolean usesClass(Class<?> c);
}

@SuppressWarnings("serial")
class Win32MediaSize extends MediaSizeName {

    public static synchronized Win32MediaSize findMediaName(String name);

    public static MediaSize[] getPredefMedia();

    public Win32MediaSize(String name, int dmPaper) {
    }

    int getDMPaper();

    protected String[] getStringTable();

    protected EnumSyntax[] getEnumValueTable();
}
